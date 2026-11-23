# Training the optional "net" expert

The cx2 entropy model works without this. The net is one extra expert: a small MLP that
predicts each EnCodec code from the previous four frames' latents. It is the only part of
the model trained by gradient descent; inference is integer-only (`../src/cx2/net.rs`), so coded
files stay bit-identical across machines. A table file (`.ecxm`) carries the net only when
`tools/build_tables.py` is given `--net`.

## What changed for the v2 retrain

| | v1 (`out/net48.pt`, replaced) | v2 (`out/net48v2.pt`, shipped) |
|---|---|---|
| data | `train_data_48k.npz`, 6.7 h | `train_data_48k_full.npz`, 24.3 h (13.14 M frames) |
| batch, steps | 512, 200,000 (28 passes over 3.6 M frames) | 2048, 180,000 (28 passes over 13.14 M frames) |
| peak learning rate | 1.5e-3 | 3e-3 (one-cycle, 5% warm-up) |
| bandwidths | 2, 4, 8, 16 | 2, 4, 8, 15, 16 (`music24` has its own embedding) |
| codebook counts per batch | 16 half the time, else uniform | 15: 35%, 16: 25%, 8: 15%, 4: 15%, 2: 10% (`--k-mix`) |
| position in segment | none | learned first-layer bias per position bucket |
| context cuts | 3% of samples at random frames | at segment starts, every 1, 2 or 4 segments (the page lengths) |
| table | version 4 | version 5: self-contained, carries its centroids |

Why these numbers:

- **Steps.** 28 passes over the set, as v1 made over its smaller one:
  28 x 13.14 M frames / 2048 = 179,700, rounded to 180,000 steps (369 M frame samples).
- **Learning rate.** Adam's noise scale grows with the square root of the batch, so 4x the
  batch takes 2x the rate: 1.5e-3 x 2 = 3e-3. The one-cycle schedule is kept; its warm-up
  (5% = 9,000 steps) is longer in samples than v1's, which helps at the higher peak. If
  `train_bits` jumps up or turns NaN in the first 10,000 steps, restart with `--lr 2e-3`.
- **Position.** The 48 kHz model codes 1 s segments of 150 frames whose first ~30 ms and last
  ~10 ms are damaged, so code statistics depend on where a frame sits. The net adds a learned
  bias to its first layer for the frame's position `t mod 150`: one bucket each for frames
  0-11 and 138-149, nine for the middle (33 buckets). The archives keep each clip whole and
  segments start every 150 frames from the clip start, so `t` = frames since the clip start
  (checked against all 1,223 source clips). The coder uses frames since the last
  reset, which is the same number because resets always fall on segment starts. The one
  exception is a file's very last segment when the one before it is short (a 149-frame
  penultimate segment occurs in 1 of 1,223 clips); both sides then still agree.
  Cost: 384 additions per frame, once per frame (about 26 per code at 15 codebooks, against
  about 70,000 multiply-adds per code for the rest of the net), plus 50 KB of table.
- **Speed.** All data (about 0.6 GB on the GPU) stays on the GPU and every batch is gathered
  there, so no NumPy runs per step; Adam is the fused CUDA one and TF32 matmuls are on
  (`--no-tf32` turns them off). TF32 and the optional `--amp` (bf16) only change training
  arithmetic; `export_net.py` measures the float/int gap with fp32 inference of the final
  weights either way. `torch.compile` is not used: it needs Triton, which PyTorch does not
  ship for Windows, and the model is small enough that kernel launches, not arithmetic, set
  the pace. `--data-on-cpu` keeps the codes in host memory and copies pinned batches, for
  small GPUs.
- **Safety.** Every 5,000 steps (`--ckpt-every`) the full state (weights, optimiser,
  schedule, both random generators, step) goes to `OUT.ckpt`; `--resume OUT.ckpt` continues
  exactly where it stopped (verified: a resumed run ends with bit-identical weights). The
  network itself (`OUT`), its description (`OUT.json`) and the progress log (`OUT.log`, one
  JSON line per evaluation, appended) are rewritten at every evaluation (`--eval-every`).

## On the training PC (Ryzen 9 7950X, RTX 3090, Windows 11)

Copy this folder (`train/` at the top of the repository; it was `entropy/train/` before the
picotone merge) with `train_data_48k_full.npz` and `train_data_48k.npz`.
Nothing else is needed: no audio, no EnCodec weights, no Rust.

Only uv and `pyproject.toml` are used (no pip, no venv by hand, no requirements file).
`pyproject.toml` takes torch from the PyTorch CUDA index through `[tool.uv.sources]` and
`[[tool.uv.index]]`; the index URL ends in the CUDA build (`cu132`), which the NVIDIA driver
must support (`nvidia-smi` shows the highest CUDA version it supports; a recent 3090 driver
is fine). Change the suffix there if needed, never install torch by hand.

```
cd train
uv sync --extra cuda
uv run --extra cuda python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

The last line must print `True NVIDIA GeForce RTX 3090`. Always pass `--extra cuda` to
`uv run`, so uv keeps the CUDA build installed.

### Step 1: the learning-curve check (about 30-60 minutes)

```
uv run --extra cuda learning_curve.py --data train_data_48k_full.npz
```

Trains on 25%, 50% and 100% of the clips (chosen by clip, deterministically; each share
contains the smaller ones) for 40,000 steps each at batch 2048, then prints held-out bits per
code at K = 15, 8 and 4 for both test sets, and the 50% -> 100% gain. A gain above ~0.02 bits
per code at K=15 means more data still helps and the full run is worth it. Each run has a
1-hour limit (`--hours-per-run`). Outputs: `out/lc_f025.pt`, `out/lc_f05.pt`, `out/lc_f10.pt`
with their `.log` and `.json`.

### Step 1b (optional, in parallel or before): the long-context test (about 30-60 minutes)

```
uv run --extra cuda train_gru.py --data train_data_48k.npz --steps 20000 --compare
```

Float only, no Rust inference yet. It trains a GRU trunk (one layer, 320 units, state
carried over chunks of 300 frames starting on segment starts and reset at each chunk start)
in front of the same per-code heads, and the MLP baseline with `train_net.py` for the same
20,000 steps (about the same number of frames per step: 7 chunks x 300 vs 2048). Both are
scored on the same held-out chunks, all frames, with the MLP's context cut at chunk starts
too. It prints bits per code at K = 4, 8, 15, 16 and the verdict; the kill rule is a gain under
0.2 bits per code at K = 15. `--arch attn` tries two causal attention layers over the last 32
frames instead. Multiply-adds per frame (printed too): MLP trunk 295 k, GRU trunk 860 k, attention
trunk 1.9 M, heads at K=15 1.04 M, so a GRU net would cost about 1.4x today's per
frame. Running it at the same time as step 1 is fine (both fit in 24 GB) but each then
runs slower. Outputs: `out/cmp_mlp.pt` (+ `.log`), `out/cmp_gru.pt`, and the printed table.

### Step 2: the full run (about 45-75 minutes)

```
uv run --extra cuda train_net.py --data train_data_48k_full.npz --out out/net48v2.pt
```

The defaults are the v2 settings above. Expected time: the v1 run (200,000 steps at batch
512) took 1.5-2 h, i.e. 27-36 ms per step, about a third of it NumPy batch assembly on one
core, which is now gone. Steps at batch 2048 are 4x the arithmetic but the 3090 is
launch-bound at this size, so expect roughly 15-25 ms per step: 45-75 minutes for 180,000
steps, plus under 5 minutes of evaluations. Every log line has `ms_per_step`; at step 2000,
`ms_per_step x 3` is the total in minutes. Memory: under 3 GB of GPU memory, under 2 GB of RAM.

What to watch: `test_old_K16` (bits per code on the old test set, 16 codebooks, all context)
is directly comparable with v1's log: 8.30 at 20,000 steps, 8.20 at 40,000, 7.87 at the end.
`test_old_K15` is `music24`'s number.

### Step 3: if it stops

A crash, a reboot or Ctrl+C loses at most 5,000 steps:

```
uv run --extra cuda train_net.py --data train_data_48k_full.npz --out out/net48v2.pt --resume out/net48v2.pt.ckpt
```

The checkpoint's settings (steps, batch, learning rate, fraction, bandwidths, sizes, seed)
override the command line, so the schedule continues unchanged. `OUT.log` is appended to.
`OUT.ckpt` is about 17 MB and is not needed afterwards.

### Step 4: what to bring back

Into `train/out/` on the development machine:

- `out/net48v2.pt`, `out/net48v2.pt.json`, `out/net48v2.pt.log` (required);
- `out/lc_f025.pt.log`, `out/lc_f05.pt.log`, `out/lc_f10.pt.log` and the printed learning-curve table;
- `out/cmp_mlp.pt.log` and the printed long-context table (and `out/cmp_gru.pt` if it won).

`out/net48.pt` (v1) stays as it is.

## Bringing it back (development machine)

From the repository root, with the files above in `train/out/`. The new table gets a new
name; the shipped tables never change in place (streams store their table's hash).

```
cargo build --release

# 1. Export check: float vs emulated-integer bits per code at K = 16, 15, 8, 4; must print OK.
#    (Also checks that the GPU batching equals the NumPy reference.)
cd train
uv run --extra cpu export_net.py --ckpt out/net48v2.pt --data train_data_48k_full.npz
cd ..

# 2. Build the lean hybrid table (version 5, centroids inside).
uv run --project tools tools/build_tables.py 48k models/cx2_encodec_48khz_net2_v5.ecxm --net train/out/net48v2.pt --reset 2 --no-hist --no-static --geo 0

# 3. Determinism and size on the preset test sets, new table then the shipped one (name the
#    new table after the next network, for example net3). Each line runs `picotone dev
#    check`: .pico pack / unpack compare, seek decode, a stream without a length field,
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

Compare the `bits/code` and `% saved` lines of step 3 between the two tables (WHITEPAPER §6.3
has the shipped `net48v2` table's numbers: music24 8.00 / 8.64, music12 7.39 / 8.23, music6
7.00 / 7.95 bits per code, old / new set). Each run also writes
`work/eval_<table>_<set>_rauto.json`. A version 5 table needs no EnCodec weights.

5. **Make it the default** if it wins: point the 48 kHz default in `src/main.rs`
   (`default_table`), `wasm/src/lib.rs`, `wasm/tools/export-models.mjs` and
   `web/tools/export-weights.mjs` (`TABLES`) at the new file, **delete the old table** from
   `models/`, re-export the browser models, and rerun the tests. picotone ships only its newest
   tables; files coded with the old table must be re-encoded. Keep the new file name: the table
   hash in old files then fails cleanly with a "wrong table" error.

## Table versions

- **Version 4** (the tables before the picotone merge, kept in `work/legacy_tables/`):
  centroids come from the EnCodec weights, rounded by `features.rs::to_fixed`; the hash covers
  the table bytes and those centroids. Its net is in format 1. `picotone dev` still loads them
  (with `--codec <weights>`), but `.pico` files are always written with version 5 tables.
  `tools/upgrade_table.py` converted each one to version 5 exactly (the same predictions), and
  `build_tables.py` with the settings above rebuilds the net table byte for byte.
- **Version 5** (all shipped tables): the fixed-point centroids the coder uses are stored in the table, right
  after the exponential table, `[codebook][code][128]` as i16 (16 x 1024 x 128 x 2 bytes =
  4 MiB at 48 kHz; the latent dimension is 128). `Model::from_table` needs nothing else, and
  the hash covers exactly what the coder computes with, so f16 or otherwise different weight
  files cannot change it. Its net, if any, is in format 2 (the position input; period 0
  turns it off, so a v1 net can be stored too). Old binaries reject version 5 cleanly.
  Size: the centroids add 4,194,304 bytes. The shipped lean hybrid is 3,158,324 bytes (2.14 MB
  gzipped); the smoke-test v5 lean hybrid is 7,405,014 bytes (4.59 MB gzipped). Storing less
  would not be exact: the geo expert is off in the lean table, but every codebook's full
  128-dimensional centroids enter the centroid sums the net reads (previous frames and the
  partial sum), at the full 13-bit precision (values within +-4095). The only exact saving is
  repacking to 13 bits per value (3.4 MB instead of 4.2 MB), which gzip already mostly gets.

## Making or extending the data archive

```
uv run --extra cpu collect_data.py --tag 48k --weights ../weights \
    --fit "../work/ecdc/fit__*.48k.ecdc" "../work/ecdc2/fit2__*.48k.ecdc" \
    --test-old "../work/ecdc/test__*.48k.ecdc" --test-new "../work/ecdc2/test2__*.48k.ecdc" \
    --out train_data_48k_full.npz
```

The inputs are encoded at 24 kbps (16 codebooks for the 48 kHz model; `--tag 24k` and 32
codebooks for the other one). The existing archive is flat `.ecdc` files (the codec's raw
codes) from the encoder before the picotone merge: `work/ecdc` holds the phase 1 clips and
`work/ecdc2` the phase 2 clips; `DATA_SOURCES.json` lists where each came from and its licence.
`collect_data.py` still reads them directly, and it also reads `.pico` files (through
`picotone dev codes`; build picotone first or set `PICOTONE_EXE`) and code-bundle `.npy`
files. The codes, and so the archive, are the same whichever container carried them (checked
on 48 kHz and 24 kHz clips packed both ways with `picotone dev pack`).

To add audio: clip it to 48 kHz stereo WAV, encode it with
`picotone encode in.wav fit3__<genre>__<name>.48k.pico --model encodec-48k -b 24 --beam 1` (no preset, so
it keeps the stock stride and 16 codebooks like the rest of the set), add it to
`DATA_SOURCES.json`, and pass it to `--fit` beside the `.ecdc` patterns. Then re-run the
command above. Test streams must stay out of `--fit`. Each clip is stored
whole (`*_lengths`), which is what the position input relies on.

Larger networks are allowed within the integer format's limits: `--h1` up to 512, `--h2`
up to 256, `--rank-low` / `--rank-high` up to 128. Cost per coded symbol grows with the head
ranks (a 1024 x rank product per code), so check speed after converting.
