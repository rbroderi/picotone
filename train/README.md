# train: the cx2 entropy-coder networks

This folder trains the two networks inside picotone's cx2 tables:

- **The EnCodec "net" expert** (`train_net.py`, `export_net.py`): one expert of the EnCodec cx2 model, in
  `models/cx2_encodec_48khz_net2_v5.ecxm`. The other experts are counted from data by
  `tools/build_tables.py`, and a table works without the net.
- **The HQ network** (`hq_net.py`): the whole cx2 model for HQ streams, `models/cx2_dac44.ecxm`.

The HQ codec's own networks (the STFT encoder, the Vocos decoder, codebooks 10–16 and the post-filter) are
trained in [`hq/`](hq/README.md).

Both nets are trained with gradient descent but run with integer-only inference in Rust (`../src/cx2/net.rs`,
`../src/cx2/hq.rs`), so coded files are bit-identical on every machine.

Only uv and `pyproject.toml` are used. `pyproject.toml` takes torch from the PyTorch CUDA index through
`[tool.uv.sources]` and `[[tool.uv.index]]`; the index URL ends in the CUDA build (`cu132`), which the NVIDIA
driver must support (`nvidia-smi` shows the highest CUDA version it supports). Change the suffix there if
needed; never install torch by hand.

```
cd train
uv sync --extra cuda
uv run --extra cuda python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

Always pass `--extra cuda` (or `--extra cpu`) to `uv run`, so uv keeps that torch build installed. Give every
long run a timeout and check on it every 10–15 minutes.

# The EnCodec net expert

A small MLP that predicts each EnCodec 48 kHz code from the previous four frames' latents and the part of the
current frame already coded. A table carries the net only when `tools/build_tables.py` is given `--net`.

## Settings

| | |
|---|---|
| data | `train_data_48k_full.npz`, 24.3 h (13.14 M frames) |
| batch, steps | 2048, 180,000 (28 passes over 13.14 M frames) |
| peak learning rate | 3e-3, one-cycle, 5% warm-up |
| bandwidths | 2, 4, 8, 15, 16 kbps (`music24` has its own embedding) |
| codebook counts per batch | 15: 35%, 16: 25%, 8: 15%, 4: 15%, 2: 10% (`--k-mix`) |
| position in segment | a learned first-layer bias per position bucket |
| context cuts | at segment starts, every 1, 2 or 4 segments (the page lengths) |

- **Steps.** 28 × 13.14 M frames / 2048 = 179,700, rounded to 180,000 steps (369 M frame samples).
- **Learning rate.** If `train_bits` jumps up or turns NaN in the first 10,000 steps, restart with `--lr 2e-3`.
- **Position.** The 48 kHz model codes 1 s segments of 150 frames whose first ~30 ms and last ~10 ms are
  damaged, so code statistics depend on where a frame sits. The net adds a learned bias to its first layer for
  the frame's position `t mod 150`: one bucket each for frames 0-11 and 138-149, nine for the middle (33
  buckets). The archive keeps each clip whole and segments start every 150 frames from the clip start, so `t` =
  frames since the clip start. The coder uses frames since the last reset, which is the same number because
  resets always fall on segment starts. Cost: 384 additions per frame (about 26 per code at 15 codebooks,
  against about 70,000 multiply-adds per code for the rest of the net), plus 50 KB of table.
- **Speed.** All data (about 0.6 GB) stays on the GPU and every batch is gathered there; Adam is the fused CUDA
  one and TF32 matmuls are on (`--no-tf32` turns them off). TF32 and the optional `--amp` (bf16) only change
  training arithmetic; `export_net.py` measures the float/int gap with fp32 inference of the final weights.
  `torch.compile` is not used: it needs Triton, which PyTorch does not ship for Windows, and at this size kernel
  launches, not arithmetic, set the pace. `--data-on-cpu` keeps the codes in host memory and copies pinned
  batches, for small GPUs. Memory: under 3 GB of GPU memory, under 2 GB of RAM.
- **Safety.** Every 5,000 steps (`--ckpt-every`) the full state (weights, optimiser, schedule, both random
  generators, step) goes to `OUT.ckpt`; `--resume OUT.ckpt` continues exactly where it stopped (a resumed run
  ends with bit-identical weights). The network (`OUT`), its description (`OUT.json`) and the progress log
  (`OUT.log`, one JSON line per evaluation, appended) are rewritten at every evaluation (`--eval-every`).

## Training

Training needs only the data archive (below): no audio, no EnCodec weights, no Rust.

```
uv run --extra cuda train_net.py --data train_data_48k_full.npz --out out/net48v3.pt
```

On an RTX 3090 a step takes roughly 15-25 ms, so 180,000 steps take 45-75 minutes plus under 5 minutes of
evaluations. Every log line has `ms_per_step`; at step 2000, `ms_per_step x 3` is the total in minutes. Watch
`test_old_K16` (bits per code on the old test set, 16 codebooks, all context); `test_old_K15` is `music24`'s
number.

If it stops, a run loses at most 5,000 steps:

```
uv run --extra cuda train_net.py --data train_data_48k_full.npz --out out/net48v3.pt --resume out/net48v3.pt.ckpt
```

The checkpoint's settings (steps, batch, learning rate, fraction, bandwidths, sizes, seed) override the command
line, so the schedule continues unchanged. `OUT.ckpt` (about 17 MB) is not needed afterwards.

Larger networks are allowed within the integer format's limits: `--h1` up to 512, `--h2` up to 256,
`--rank-low` / `--rank-high` up to 128. Cost per coded symbol grows with the head ranks (a 1024 × rank product
per code), so check speed after converting.

## Making a table from it

From the repository root, with the network in `train/out/`. A new table gets a new name (streams store their
table's hash).

```
cargo build --release

# 1. Export check: float vs emulated-integer bits per code at K = 16, 15, 8, 4; must print OK.
#    (Also checks that the GPU batching equals the NumPy reference.)
cd train
uv run --extra cpu export_net.py --ckpt out/net48v3.pt --data train_data_48k_full.npz
cd ..

# 2. Build the lean hybrid table.
uv run --project tools tools/build_tables.py 48k models/cx2_encodec_48khz_net3_v5.ecxm --net train/out/net48v3.pt --reset 2 --no-hist --no-static --geo 0

# 3. Determinism and size on the preset test sets, new table then the shipped one. Each line runs
#    `picotone dev check`: .pico pack / unpack compare, seek decode, a stream without a length field,
#    baseline (non-SIMD) path compare; it fails on any differing code.
foreach ($t in "cx2_encodec_48khz_net3_v5", "cx2_encodec_48khz_net2_v5") {
  foreach ($s in "ps_music24_old 15", "ps_music24_new 15", "ps_music12_old 8", "ps_music12_new 8", "ps_music6_old 4", "ps_music6_new 4") {
    $a = $s.Split(" ")
    uv run --project tools tools/check_testset.py 48k models/$t.ecxm auto $a[0] $a[1]
  }
}

# 4. Rust checks (tests/pico.rs covers the format with the shipped tables).
cargo fmt --check
cargo clippy --all-targets
cargo test --release
```

Compare the `bits/code` and `% saved` lines of step 3 between the two tables (WHITEPAPER §6.3 has the shipped
table's numbers: music24 8.00 / 8.64, music12 7.39 / 8.23, music6 7.00 / 7.95 bits per code, old / new set).
Each run also writes `work/eval_<table>_<set>_rauto.json`.

**Make it the default** if it wins: point the 48 kHz default in `src/main.rs` (`default_table`),
`wasm/src/lib.rs`, `wasm/tools/export-models.mjs` and `web/tools/export-weights.mjs` (`TABLES`) at the new
file, delete the old table from `models/`, re-export the browser models, rerun the tests, and re-encode every
file coded with the old table. A file coded with a table picotone does not ship fails cleanly with a "wrong
table" error, since its table hash matches none.

## The table format

All tables are version 5. The fixed-point centroids the coder uses are stored in the table, right after the
exponential table, `[codebook][code][128]` as i16 (16 × 1024 × 128 × 2 bytes = 4 MiB at 48 kHz; the latent
dimension is 128). `Model::from_table` needs nothing else, and the hash covers exactly what the coder computes
with, so f16 or otherwise different weight files cannot change it. The net is in format 2 (with the position
input; period 0 turns it off). The shipped lean hybrid is 3,158,324 bytes (2.14 MB gzipped). Storing less
would not be exact: the geo expert is off in the lean table, but every codebook's full 128-dimensional
centroids enter the centroid sums the net reads (previous frames and the partial sum), at the full 13-bit
precision (values within ±4095). The only exact saving is repacking to 13 bits per value (3.4 MB, against
4.2 MB), which gzip already mostly gets.

## The data archive

```
uv run --extra cpu collect_data.py --tag 48k --weights ../weights \
    --fit "<codes>/fit__*.48k.pico" --test-old "<codes>/test__*.48k.pico" --test-new "<codes>/test2__*.48k.pico" \
    --out train_data_48k_full.npz
```

The inputs are the codes of a folder of 48 kHz stereo WAV clips you provide, from the public-domain and CC0
sources listed in `../DATA_SOURCES.json`, encoded at 24 kbps with no preset (16 codebooks and the stock stride;
`--tag 24k` and 32 codebooks for the 24 kHz model):

```
picotone encode in.wav fit__<genre>__<name>.48k.pico --model encodec-48k -b 24 --beam 1
```

`collect_data.py` reads `.pico` files (through `picotone dev codes`; build picotone first or set
`PICOTONE_EXE`), code-bundle `.npy` files, and flat EnCodec code files (`.ecdc`, read by `tools/ecdcio.py`);
the codes, and so the archive, are the same whichever container carried them. Test streams must stay out of
`--fit`. Each clip is stored whole (`*_lengths`), which the position input relies on. List every new source
in `DATA_SOURCES.json`.

# The HQ table (`models/cx2_dac44.ecxm`)

HQ streams (DAC 44 kHz codes) are coded by their own cx2 model: one integer network, no other experts
(`../src/cx2/hq.rs`). `hq_net.py` trains it and writes its table. The table depends only on the codes it is
trained on and on DAC's 9 × 1024 codebook vectors (`quantK.codebook_unit` in `dac44_quantizer.safetensors`, a
fixed half of each code's embedding). So a new encoder that keeps those codebooks needs only the three steps
below.

The fit set is every clip of the `fit` and `fit2` splits except the clips in `EXCLUDE` in `tools/dac_codes.py`
(six clipped, distorted DTRASH masters and two clips of digital silence). The test set is the 23 clips of the
`test` split. The clips are a folder of WAVs you provide, from the sources in `../DATA_SOURCES.json`, with a
JSON list per split (`[{name, split}]`). No fit clip shares a source file or URL with a test clip; some share
an album (other Goldberg variations, Chopin pieces and Brahms symphonies).

```sh
# 1. Dump codes: per clip one [frames, 16] .npy per coded channel and a .json sidecar.
#    Runs `picotone encode --dump-codes` at 16 codebooks (no cx2 coding, so any table loads);
#    skips clips already dumped (about 3-7 s per clip). Point PICOTONE_EXE at a copy of the
#    binary if you rebuild picotone meanwhile.
python ../tools/dac_codes.py --list <clips>/clips_fit.json --split fit --wav <clips>/wav \
    --out ../work/dac_codes/fit --threads 4 --jobs 4
python ../tools/dac_codes.py --list <clips>/clips2.json --split fit2 --wav <clips>/wav2 \
    --out ../work/dac_codes/fit2 --threads 4 --jobs 4
python ../tools/dac_codes.py --list <clips>/clips_test.json --split test --wav <clips>/wav \
    --out ../work/dac_codes/test --threads 4 --jobs 4

# 2. Train: tiers mixed per channel (q1-q3 6% each, q4-q6 10%, q9 22%, q11-q16 5%; never q7,
#    q8 or q10), 80-frame pages, 8 frames of history. Prints bits per code per tier and per role
#    on the test codes. On a laptop GPU (--device cuda) 80,000 steps take about 9 minutes; on
#    the CPU about 1.2 s per step at 16 codebooks, so use the GPU.
uv run --extra cpu python hq_net.py train --fit ../work/dac_codes/fit ../work/dac_codes/fit2 \
    --test ../work/dac_codes/test --weights ../weights --out out/hq_net.pt --steps 80000 --device cuda

# 3. Export: the table, the integer-vs-float check per tier (coded bits, through the coder's
#    frequency table), and the Rust parity fixture.
uv run --extra cpu python hq_net.py export --ckpt out/hq_net.pt --test ../work/dac_codes/test \
    --weights ../weights --table ../models/cx2_dac44.ecxm --fixture ../tests/data/cx2_dac44_fixture.json
```

Then `cargo test --release --test hq_cx2` checks Rust against the fixture: the logits on the AVX2 and portable
paths, page round trips, and truncation. `picotone --threads 1 dev dac-pack models/cx2_dac44.ecxm
../work/dac_codes/test` reports the real coded size per tier and the cx2 time. A new table changes the table
hash, so HQ files coded with another table must be re-encoded; ship the new table only.

What the network sees: the channel's previous 8 frames *in the same page* (pages are 80 frames, and every page
and channel starts empty), the codebooks already coded in this frame, and the channel's role (left/right, mid
or mono, side). Choices measured on the 23 test clips:

- **History.** 4 frames costs 0.03 bits per code more than 8, and 16 frames buys nothing.
- **Other channel.** Coding the other stereo channel as an input saves under 0.01 bits.
- **Page length.** Longer pages save under 0.03 bits.
- **Additive experts** (codebook pairs, the codebook geometry): under 0.006 bits together.
- **Fit data and steps** (STFT encoder codes, float net, bits per code at q9 / q3 on the test codes; "other" is
  the 13 test clips with no album in the fit set):

  | fit set | steps | all 23 clips | other 13 |
  |---|---|---|---|
  | 407 clips, 16.8 channel-hours | 40k | 7.984 / 7.100 | 8.283 / 7.405 |
  | 407 clips | 80k | 7.893 / 7.060 | 8.204 / 7.372 |
  | all fit and fit2: 1,110 clips, 45.3 channel-hours | 40k | 8.018 / 7.111 | 8.283 / 7.388 |
  | all fit and fit2 | 80k | 7.909 / 7.054 | 8.188 / 7.338 |
  | all fit and fit2, 16 codebooks (shipped) | 80k | 7.854 / 7.125 | |

  Steps matter more than data here. The 407-clip set wins at q9 on all 23 clips only through the four Goldberg
  test clips (0.13-0.23 bits each): 27 of its 407 clips are other variations from the same recording. On the
  other 13 clips the larger set wins at both q9 and q3, so the table uses all 1,110 clips.
- **16 codebooks.** Against a 9-codebook table on the same clips, the 16-codebook table saves 0.055 bits at
  q9 and costs 0.071 at q3, since q1-q3 get less of the tier mix. Real coded size on the 23 test clips
  (`dev dac-pack`), kbps per channel: q1 0.60, q3 1.84, q6 3.94, q9 6.09, q10 6.56, q11 7.32, q12 8.06,
  q13 8.38, q14 9.13, q15 9.87, q16 10.57. Codebooks 10 and 13 are cheap (5.6 and 3.8 bits per code at q16:
  they use few of their entries).
- **Head rank.** Rank 64 costs 0.085 bits at q9 and 0.048 at q3 against rank 128, and decodes only about 1.2x
  faster. The table uses rank 128.

The integer table codes within 0.002 bits per code of the float network.
