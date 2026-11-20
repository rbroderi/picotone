# distill: a 512-wide student of DAC's decoder

This folder trains a narrow copy of the **DAC 44 kHz** decoder. The goal is DAC-like quality per bit that
decodes faster than real time in single-thread WebAssembly, which picotone's consumer (gaim) needs.

- **The student.** DAC's decoder architecture with a width of 512 (stock is 1536). It halves the width per
  stage, keeps the same upsampling rates (8, 8, 4, 2) and the same residual units (dilations 1/3/9), and uses
  Snake activations. Use `--width 384` for the smaller one.
- **The inputs.** The frozen DAC encoder and quantiser (`descript-audio-codec` 1.0.0 weights). The default is
  5 codebooks per channel, which is 8.6 kbps stereo before entropy coding.
- **The losses.** DAC's own recipe, scored against the **original audio**. Matching the stock decoder's output
  is optional.

Nothing here has been trained yet. The code was checked with a CPU smoke run (see the end of this file).

## Cost: measured, per second of audio, one channel

Run `uv run --extra cuda python report_model.py` to reproduce this table. The parameter count and MACs come from forward
hooks.

| Decoder | Params | GMAC/s | Snake evals/s | MAC-eq ratio | Projected wasm, 1 thread, stereo | Projected WebGPU, stereo |
|---|---|---|---|---|---|---|
| Stock DAC (1536) | 54.09 M | 69.26 | 77.9 M | 1.00× | 0.22× real time | 0.78× |
| **Student 512** | **8.46 M** | **7.91** | 26.0 M | **8.7×** | **about 1.9×** | **about 6.8×** |
| Student 384 | 5.45 M | 4.51 | 19.5 M | 15.2× | about 3.3× | about 11.8× |

- **How the projections are made.** They scale the earlier study's stock-DAC browser figures by the
  MAC-equivalent ratio. Snake counts as 4 MAC-eq, as a degree-3 polynomial with range reduction
  (`snake_bench`).
- **Stereo** is two mono decodes.
- **The 512 student is marginal for "faster than real time" in single-thread wasm** (about 1.9×). It has
  headroom on WebGPU and threaded wasm. The 384 student is the fallback if 512 is too slow in practice; it has
  a bigger quality risk.

## Files

| File | Purpose |
|---|---|
| `student.py` | The student decoder, Snake (custom autograd), weight norm with a fold-out helper, the `--init slice` warm start, and the MAC counter |
| `teacher.py` | Frozen DAC: crop encoding, a codes → z_q helper with per-item codebook counts, and the stock decoder |
| `losses.py` | Multi-scale mel loss, the MPD and MRD discriminators, LS or hinge GAN loss, and feature matching (plain torch, no audiotools) |
| `metrics.py` | Log-mel distance in dB, MR-STFT, SNR and SI-SDR. Same definitions as the earlier picotone/DAC comparison |
| `audio_io.py` | Detects the codec from the file's bytes (Ogg Vorbis, Ogg FLAC, Opus, FLAC, MP3, WAV), reads zip members, and resamples to 44.1 kHz |
| `corpus.py` | Licence classes and presets, the bandwidth probe, the quality judgement, the split by artist, and the data reports |
| `prep_pd.py` | Builds the manifest for picotone's existing public-domain/CC0/CC-BY corpus, plus the 53 picotone test clips as the eval set |
| `prep_corpus.py` | Builds a manifest for any folder of audio (recursive, zips included), for example the Jamendo dump |
| `data.py` | The streaming loader. Each batch is a pure function of (seed, step), so a resumed run continues the same sequence |
| `train.py` | The training loop, the periodic eval, listening WAVs, the pass/fail verdict, checkpoints and `--resume`, and `--bench` |
| `verdict.py` | The pilot pass/fail rule and the full-run target |
| `compare_picotone.py` | Scores picotone music24/12/6 on the same clips and prints the comparison table with bitrates |
| `export.py` | Folds weight norm and writes the generator only, a config, a credits list and a licence notice |
| `smoke.py` | The smoke test (CPU here; it can also run on the 3090 with `--device cuda`) |
| `data/picotone_testclips.json` | The 53 clips of picotone's music24 "old" and "new" test sets |
| `data/picotone_eval.json` | Picotone's scores on those clips (precomputed here; `compare_picotone.py` reuses it) |

## 1. Setup on the 3090 PC

Install ffmpeg and put it on PATH (`winget install Gyan.FFmpeg`). Ogg FLAC needs it, because libsndfile can't
read Ogg FLAC. Then:

```powershell
cd <repo>\distill
uv sync --extra cuda          # torch 2.15 + cu132 (change the index suffix in pyproject.toml if the driver needs it)
uv run --extra cuda python -c "import torch; print(torch.cuda.get_device_name(0))"
uv run --extra cuda python report_model.py
```

**Always run with `--extra cuda`.** A plain `uv run` re-syncs the environment without the extra, and because
descript-audio-codec depends on torch, uv swaps in the CPU-only torch from PyPI ("Torch not compiled with CUDA
enabled"). Every command below uses `uv run --extra cuda python ...`.

The first run downloads the DAC weights (about 300 MB) into `~/.cache/descript`.

## 2. Data

There are two sources. Both go through the **same quality filter**, which is a test of effective bandwidth
(below). Both hold out creators and artists, and the licence policy is applied at train time (`--licences`),
so changing it doesn't need a rescan.

### 2a. The existing picotone corpus (the `pd` source)

On the build machine it lives in the session scratchpad, not in `work\`, which holds only codes:

```text
C:\Users\richa\AppData\Local\Temp\claude\8979288677f01101\7927348c-78e9-4ae8-849c-648d18ca0651\scratchpad\
    dl\manifest.json, dl\manifest2.json, dl\manifest_oga.json      (licence + creator per download, 0.6 MB)
    work\clips_fit.json, work\clips_test.json, work\clips2.json    (clip lists)
    work\wav\  (152 clips, 2.8 GB)   work\wav2\ (1,171 clips, 17.0 GB)   48 kHz stereo WAV
```

**Copy these, about 20 GB, keeping the `dl\` and `work\` layout** under one folder on the 3090 PC, for
example `D:\picotone_pd`. The scratchpad is temporary, so copy it soon. You don't need the 10.5 GB of
original downloads in `dl\`.

To re-fetch instead, run the picotone tools from `<repo>\tools`. Allow a few hours; they need ffmpeg.

```powershell
uv run python fetch_audio.py D:\picotone_pd\dl
uv run python fetch_audio_phase2.py D:\picotone_pd\dl
uv run python fetch_opengameart.py D:\picotone_pd\dl
uv run python prep_clips_phase2.py D:\picotone_pd\dl D:\picotone_pd\work
uv run python prep_clips.py D:\picotone_pd\dl D:\picotone_pd\work <encodec-exe> <weights> fit
uv run python prep_clips.py D:\picotone_pd\dl D:\picotone_pd\work <encodec-exe> <weights> test
```

`prep_clips.py` also EnCodec-encodes; its WAVs are all `distill` needs.

Then:

```powershell
uv run --extra cuda python prep_pd.py --pd-root D:\picotone_pd --out data\pd.jsonl --workers 14    # < 1 min
```

Measured on the build machine (`--min-bandwidth 18000`):

| | Clips | Hours |
|---|---|---|
| All clips | 1,303 | 28.2 |
| **Train, passing the filter** | **797** | **17.3** (10.6 CC-BY, 6.5 CC0, 0.2 PD) |
| Train, failing the filter (band-limited MP3s, mostly 16 kHz low-pass) | 251 | 6.7 |
| Held out: the picotone test sets | 185 | 3.1 |
| Excluded: creators who also appear in a test set | 70 | 1.1 |

Speech clips nearly all fail (12 kHz bandwidth). Pass rates for the other genres run from 48% (band) to
100% (orchestra).

**The licence note matters.** `DATA_SOURCES.json` calls this corpus "public domain, CC0 or CC-BY", and
13.9 h of it is CC-BY. Under the original public-domain/CC0-only rule (`--licences strict`), only **6.7 h**
passes.

### 2b. Jamendo (`jamendo_complete_zip_2012_05_20`)

From the archive.org file listing (nothing was downloaded):

- 5,405 zips, 328.7 GB, one zip per artist: `<Artist>.zip :: <Artist>/<Album>/<NN>-<Title>.ogg`.
- Each album folder also holds `cover-*.jpg`, a `license.txt` (about 200 bytes) and
  `readme-www.jamendo.com_.txt`.
- The item is also available as a torrent.
- **The tracks are `.ogg`.** A sampled track of 8.3 MB suggests Vorbis at roughly 250 kbps, not FLAC; a
  1,411 kbps reading is the decoded PCM rate. The codec is detected per file from its bytes, and Ogg FLAC
  and Opus are also handled.
- Expect roughly 2,500–3,500 h before filtering.

Keep the zips as they are, ideally on an SSD.

```powershell
uv run --extra cuda python prep_corpus.py --root D:\jamendo --source jamendo --out data\jamendo.jsonl --workers 14 `
    --url-base https://archive.org/download/jamendo_complete_zip_2012_05_20
```

- **Time.** The scan reads each track once and decodes three 8 s windows. Expect 0.5–1.5 h; the limit is the
  disk, with 329 GB to read.
- **Resumable.** Re-running it skips tracks it has already done.

What it records per track:

| Field | Source |
|---|---|
| Codec, sample rate, channels, duration | The file itself (codec from its bytes) |
| Container bitrate, zip storage mode | The file itself |
| Bandwidth | Measured |
| Artist, album, title | Tags, falling back to the path |
| Licence class and URL | Tags, falling back to `license.txt` / readme |
| Attribution URL | Tags, falling back to the readme, then `--url-base` |
| Keep or reject, with reasons and flags | The quality filter |
| Split | `train`, or `heldout` for a deterministic 2% of artists (`--heldout-frac`) |

The report (`data\jamendo.jsonl.report.json`) lists hours per split × pass/fail, per codec, sample rate,
channel count, zip storage mode, licence, reject reason and flags. `data\jamendo.jsonl.eval.json` holds 30
fixed held-out excerpts, one per artist, for evaluation.

- **Quality filter (main test).** `--min-bandwidth 18000` Hz. The effective bandwidth is the highest frequency
  where the smoothed long-term spectrum is within 60 dB (`--bw-drop-db`) of its 1–4 kHz level. It is measured
  on three windows and the median is taken. A lossless or high-bitrate file transcoded from a 128 kbps MP3
  still shows its 16 kHz cliff, so this catches what a bitrate test can't.
- **Bitrate (flag only).** `--min-kbps 160` flags lossy files below it. `--drop-low-kbps` turns the flag into a
  reject.
- **Licence policy (your choice).** `train.py --licences`, with these presets:

  | Preset | Classes |
  |---|---|
  | `strict` | `pd,cc0` |
  | `permissive` | `pd,cc0,by`: attribution only |
  | **`sharealike`** (default) | `pd,cc0,by,by-sa,fal`: adds CC BY-SA and the Free Art Licence (FAL is copyleft like BY-SA, so it is included) |
  | `commercial` | adds `by-nd` |
  | `all` | everything |

  You can also give a comma list of classes, for example `--licences pd,cc0,by,by-sa` to leave FAL out. NC
  licences are never in a preset. Jamendo in 2012 was mostly BY-SA and BY-NC-*, so the BY-SA default keeps much
  more of it than `permissive` would. Check `licence_passing` in the report before choosing. Whatever the
  policy, every admitted track is credited (see Licences, below).
- **If zips are deflated** (see `zip_member` in the report), each training window reads the whole member, which
  is about 8 MB. On a hard disk, extract the zips first; `prep_corpus` scans folders just as well. Stored
  members are seeked in place.

### Mixing (`--mix`)

- **The default, `--mix auto`,** samples each source by its duration, but no source gets less than
  `--mix-floor 0.2`.
- **Why 0.2.** At about 1,500 h of Jamendo, duration alone would give the 17 h set about 1%. That set is the
  closest to gaim's material (game music, SFX, orchestral, piano) and shares genres with the picotone test
  sets.
  - At 20%, a 400k × 32-crop run draws about 260 h of crops from its 35 channel-hours, about 8 passes. That
    is mild for a decoder conditioned on codes.
  - Much higher, and the discriminator starts to memorise the set.
  - With `--licences strict` (6.7 h), consider `--mix-floor 0.1`.
- **Explicit weights** look like `--mix pd=0.3,jamendo=0.7`.

## 3. Picotone baseline on the same clips

`data/picotone_eval.json` was computed on the build machine with
`C:\Users\richa\workspace_local\encodecx\target\release\picotone.exe` (presets music24/12/6), so this step is
optional on the 3090 PC.

Measured on the 53 clips (10 s each, scored at 44.1 kHz against the same reference the trainer uses; log-mel
in dB, lower is better):

| Codec | kbps | old: log-mel | old: SI-SDR | new: log-mel | new: SI-SDR | all: log-mel | all: MR-STFT | all: SI-SDR |
|---|---|---|---|---|---|---|---|---|
| picotone music24 | 19.87 | 3.065 | 11.02 | 2.524 | 9.75 | 2.759 | 1.259 | 10.30 |
| picotone music12 | 9.97 | 3.350 | 9.16 | 2.812 | 8.05 | 3.046 | 1.315 | 8.53 |
| picotone music6 | 4.80 | 3.760 | 6.50 | 3.221 | 5.73 | 3.455 | 1.402 | 6.07 |
| **stock DAC q5** (the teacher) | 8.61 | 2.837 | 7.59 | 2.776 | 5.81 | 2.803 | 1.258 | 6.58 |

What this shows:

- **At a matched bitrate, stock DAC q5 beats picotone on log-mel:** 2.80 dB at 8.6 kbps against music12's
  3.05 dB at 10.0 kbps, and roughly matches music24 (2.76 dB) at 43% of its bitrate. On the old set it beats
  music24 (2.84 vs 3.07); on the new set it doesn't (2.78 vs 2.52).
- **On SI-SDR, DAC is worse** (6.6 dB against 8.5 for music12). A GAN decoder doesn't keep waveform phase
  closely. That is why the verdict compares SI-SDR only with stock DAC, not with picotone.
- **So the student's bar is "close to stock DAC", and "better than music12" at the same bitrate.** "Better
  than music24" would fail the teacher itself.
- Stock DAC was scored here with `distill`'s own pipeline (fp32, CPU, the 53 clips at 44.1 kHz).
  `train.py` repeats it on the 3090 and writes `stock_eval.json`.

The DAC rates are raw: 10 bits per code, 86.13 frames/s, two channels, before any entropy coding.

To recompute it, copy `picotone.exe` along with `weights\` and `models\`, delete the json, and run:

```powershell
uv run --extra cuda python compare_picotone.py --eval-set data\pd.jsonl.eval.json --exe <path>\picotone.exe `
    --out data\picotone_eval.json
```

At any time, this prints the table of picotone, stock DAC and the student, with bitrates:

```powershell
uv run --extra cuda python compare_picotone.py --eval-set data\pd.jsonl.eval.json --out data\picotone_eval.json --run runs\w512
```

## 4. Bench, pilot, full run

```powershell
# 0) smoke on the GPU (5 min)
uv run --extra cuda python smoke.py --manifest data\pd.jsonl --out runs\smoke --device cuda --steps 200 --batch 8 --seg-frames 32 --threads 8

# 1) pick the batch size (5 min): ms/step and peak memory for each size
uv run --extra cuda python train.py --run runs\bench --bench 16,24,32,48,64 --manifest data\pd.jsonl --manifest data\jamendo.jsonl --workers 12

# 2) pilot: 50k steps
uv run --extra cuda python train.py --run runs\w512 --manifest data\pd.jsonl --manifest data\jamendo.jsonl `
    --eval-set data\pd.jsonl.eval.json --eval-set data\jamendo.jsonl.eval.json `
    --picotone-json data\picotone_eval.json --width 512 --n-q 5 --batch 32 --workers 12 --steps 50000

# 3) full run: the SAME run continued (the pilot is its first 50k steps)
uv run --extra cuda python train.py --run runs\w512 <same arguments> --steps 400000 --resume

# after a crash or reboot: the same command with --resume continues from the last checkpoint (every 2,500 steps)
```

- **Without Jamendo,** drop its `--manifest` and `--eval-set`.
- **For a decoder that serves several bitrates,** use `--n-q 4,5,6,9` (quantiser dropout). The eval then
  scores every count, and the verdict uses the first.
- **The study suggests one decoder per bitrate wins.** Train the 5-codebook student first.
- **Options:**
  - `--w-teacher-mel 5`: output matching against stock DAC (off by default).
  - `--init slice`: warm start (see the design notes).
  - `--gan hinge`: the paper's hinge GAN loss.
- **Watch** `log.txt`:
  - smoothed `mel`, `fm`, `adv` and `d` losses, and gradient norms;
  - `ms/step` and `data wait` (if the data wait is more than about 10% of the step, raise `--workers`);
  - every 5,000 steps, the eval and the verdict;
  - `listen\<clip>__s<step>_q5.wav` next to `__0_original.wav` and `__0_stockDAC_q5.wav`.

### Time estimates, and the arithmetic behind them

These are per batch item per step, at the defaults (0.37 s loss segment, 4,096 samples of encoder context
each side, 2 frames of decoder context each side). The MACs are measured; the factors are standard.

| Part | Calculation | GMAC |
|---|---|---|
| Encoder, frozen | 30.6 GMAC/s × 0.557 s | 17 |
| Student, forward + backward | 3 × 7.91 × 0.418 s | 10 |
| Discriminator, about 10 forward-equivalents per step | 10 × 32.0 × 0.372 s | 119 |
| **Total** | | **≈ 146 GMAC = 0.29 TFLOP per item** |

**The discriminator is 80% of the cost.** The student's width barely affects the step time.

- **CPU smoke calibration.** At half-length segments (76 GMAC per item) the marginal cost was 1.0 s per item,
  so about 75 GMAC/s on 6 laptop threads.
- **3090 estimate.** bf16 tensor cores peak at 71 TFLOPS dense. With these conv shapes, assume 25–40%
  utilisation, which is 9–14 TMAC/s, with the fp32/TF32 encoder at about 6 TMAC/s, plus about 20% for
  elementwise and STFT work.
- **That gives about 15–20 ms per item, so batch 32 ≈ 0.45–0.65 s per step.** `--bench` gives the real figure.
- **Memory.** The smoke counted about 0.26 GiB of saved activations per item at half-length segments (fp32,
  D and G graphs summed). That is about 0.5 GiB per item at full length in fp32, and about 0.3 GiB in bf16.
  - Batch 32 needs about 10–12 GiB, plus about 1.5 GiB of weights and Adam state.
  - Batch 48 should fit in 24 GB. Take the largest size from `--bench` that peaks under about 20 GiB.
  - DAC's own recipe was 72 × 0.38 s. If 48 fits and costs under about 1.4× per step, use it.

| Stage | Steps | Time at about 0.55 s/step | Range |
|---|---|---|---|
| Setup, data prep, bench | | 1–2 h, most of it the Jamendo scan | |
| **Pilot** | 50k | **about 7.5 h** (overnight) | 5.5–11 h |
| **Full run** | 400k | **about 2.5 days** | 2–3.7 days |
| Eval overhead (every 5k steps) | | under 1 min each | |

Checkpoints are about 0.6 GB: 51M parameters (G 8.5M + D 42.5M) × 3 for the weights and two Adam moments, in
fp32. Every 25k steps one is also kept as `ckpt_<step>.pt`.

**The step count with more data.** DAC's 400k steps × 72 × 0.38 s saw about 3,000 h of crops. Here,
400k × 32 × 0.37 s is about 1,300 h of single-channel crops. That is under one pass over a 1,500 h Jamendo
set, so the step count is set by GAN convergence, not by the data.

Recommendation:

- Plan for **400k**.
- If the verdict's trend line is still falling at 400k, extend to **600k** with `--resume --steps 600000`,
  about +1.3 days.
- The extra data helps generalisation; it is not a reason to train longer.
- With the 17 h set alone, 400k steps means about 40 passes, which risks a memorising discriminator. Prefer
  having Jamendo.

## 5. The pilot's pass/fail rule

`train.py` prints a verdict after every eval. It is scored on the picotone test clips: old (23) and new (30)
sets, 10 s each, both channels. The logic is in `verdict.py`.

```text
PILOT VERDICT at 5 codebooks: PASS | BORDERLINE | FAIL
  log-mel  <= stock DAC q5 x 1.08
  SI-SDR   >= stock DAC q5 - 1.5 dB
  log-mel  <  picotone music12 on the same clips (bitrate-matched; music24 and music6 printed for information)
  trend    log-mel still falling (last eval vs the one at half the steps)
  + listening: no metallic/ringing tones, buzz, pre-echo or GAN hiss (your call)
```

With the numbers measured here (old/new mean):

| | log-mel (dB) | SI-SDR (dB) |
|---|---|---|
| Stock DAC q5 | 2.807 | 6.70 |
| **Pilot limit** | **≤ 3.031** | **≥ 5.20** |
| Full-run limit | ≤ 2.891 | ≥ 6.00 |
| picotone music12 (bitrate-matched) | must be below 3.081 | |
| picotone music24 (information only) | 2.795 | |

The pilot's log-mel limit is already below music12, so in practice the picotone clause decides only in a
borderline case. The rule compares with music12, not music24, because stock DAC q5 itself does not beat
music24 on these clips.

**Where X comes from.** The DAC paper's 512-wide model, jointly trained for 400k steps, lost **+1.8% mel
distance** (1.09 → 1.11) and **−0.4 dB SI-SDR** (9.1 → 8.7) against the 1536-wide one.

- **Full-run target:** log-mel ≤ ×1.03 and SI-SDR ≥ −0.7 dB. That is the paper's gaps plus measurement slack
  on 53 clips; it is used automatically after `--pilot-steps`.
- **Pilot:** at 50k of 400k steps, a GAN decoder's mel loss is still a few per cent above its final value. So
  the pilot allows about 2.5× the final gap (×1.08) and −1.5 dB SI-SDR.
- **Outcomes:**
  - **PASS** (all of the above, and clean listening): continue to the full run.
  - **BORDERLINE** (within 1.5× the margins): continue, and judge again at 150k.
  - **FAIL:** stop. Try width 640, `--w-teacher-mel 5`, or a longer pilot before spending 2.5 days.

## 6. What to bring back

- `runs\w512\`:
  - `log.txt`, `train.jsonl`, `evals.jsonl`, `verdicts.jsonl`, `stock_eval.json`;
  - `listen\`, the WAVs of the latest and 2–3 earlier evals;
  - `ckpt_last.pt`, or better, the export below.
- `data\pd.jsonl.report.json`, `data\jamendo.jsonl.report.json` and the `--bench` lines.
- The export:

  ```powershell
  uv run --extra cuda python export.py --ckpt runs\w512\ckpt_last.pt --out export\student_w512_q5 --manifest data\pd.jsonl --manifest data\jamendo.jsonl
  ```

  It writes `student_w512_q5.pt` (34 MB fp32, weight norm folded) and `.json`, plus:

  - `.credits.jsonl`: artist, title, album, licence, licence URL and source URL for every training track
    admitted under the run's `--licences`, BY-SA included;
  - `.NOTICE.txt`: the DAC MIT notice, the track count per licence and, when BY-SA or FAL audio was used, the
    ShareAlike note.

  Ship all four files together.
- The output of `compare_picotone.py ... --run runs\w512`.

## 7. Design notes

- **Latents are computed on the fly, not precomputed.** The frozen encoder costs 12% of a step (17 of 146
  GMAC per item).
  - Precomputing all of Jamendo would cost about one encoder pass over 3,000 channel-hours: about 3.3e17 MAC,
    or 15–20 GPU-hours. That is more than the on-the-fly cost over a whole 400k run (2.2e17).
  - The codes would take 11.2 MB per stereo hour (9 codebooks × uint16), so 17–35 GB, or 60% of that
    bit-packed.
  - Precomputed codes would also fix crops to the frame grid. On the fly, crops start at any sample.
  - For the 17 h set alone, precomputing would be cheap (190 MB, minutes) but gains only 12% per step. It is
    not implemented.
  - Each crop is encoded with 4,096 samples of context on both sides. Checked: its codes equal whole-file
    codes 100%, on all 9 codebooks.
- **Alignment.** DAC's decoder output for frame f covers samples [512f, 512f+512) exactly. Checked: SNR peaks
  at shift 0, and ±64 samples is 4.6 dB worse.
- **Channels.** L and R are coded separately: each crop takes a random channel.
- **Codebooks.** The count is fixed (`--n-q 5`) or drawn per item from a set (dropout). The quantised latent
  is the masked sum of the codebooks' `out_proj` outputs.
- **Recipe.**
  - Losses: mel × 15 (windows 32–2048, hop ¼, 5–320 mels, log10, clamp 1e-5), feature matching × 2,
    adversarial × 1.
  - Discriminators: MPD with periods 2/3/5/7/11, and MRD at 2048/1024/512 with 5 bands.
  - Optimisers: AdamW, lr 1e-4, betas (0.8, 0.99), ×0.999996 per step, separate for G and D.
  - Gradient clipping: 1,000 for G, 10 for D.
  - DAC's released code uses a **least-squares** GAN loss (the paper text says hinge). The default follows
    the code that trained the weights; `--gan hinge` is available.
  - The discriminator is trained from scratch, because the DAC checkpoint has none.
- **AMP (bf16, CUDA only).** bf16 has fp32's exponent range, so no loss scaler is needed, and GAN training
  with bf16 autocast is routine. To be safe:
  - Snake, the residual sums, the tanh output, all STFTs, the mel loss and the GAN losses stay fp32.
  - Only convolutions run in bf16.
  - The frozen encoder runs in fp32, with TF32 allowed.
  - If losses go NaN, use `--amp off` (fp32 with TF32). Expect it to be roughly 1.5–2× slower.
- **Snake** is a custom autograd function that saves only its input, in the input's own dtype. Plain autograd
  would keep four fp32 tensors. It was gradchecked in float64.
- **Warm start (`--init slice`) is implemented but off by default.** It keeps each stage's top-energy teacher
  channels; slicing commutes with Snake, while PCA would mix channels and break Snake's per-channel form.
  Measured on a real clip at step 0: its mel loss is 9.4 against 5.8 for random init (stock: 1.55). The
  dropped 2/3 of each residual stream leaves a badly scaled network. A 2k-step A/B on the 3090 would settle
  whether it trains faster; it isn't expected to.
- **Resume is bit-exact.** Each batch depends only on (seed, step), and models, optimisers, schedulers and RNGs
  are restored.
  - Checked on CPU: 20 steps straight and 10 + resume 10 give the same weight hash.
  - On CUDA, cuDNN's non-deterministic kernels make the weights drift by float rounding, but the data
    sequence is still exact.

## 8. If it passes: integration (not built)

| Piece | What | Effort |
|---|---|---|
| Rust student decoder in picotone | Conv1d (dilated), polyphase ConvTranspose (both exist for EnCodec), and Snake as a degree-3 polynomial with range reduction (`snake_bench`, +0.005 dB log-mel). Weight norm folded at export (`export.py`); safetensors loader. Fully convolutional, so chunked decode with a receptive-field overlap replaces the segment/LSTM logic. Parity test against PyTorch. | 1–1.5 weeks |
| wasm | The same crate; the existing packed-panel GEMM (SIMD128 / relaxed SIMD) and the worker plumbing carry over. The work is benchmarking single-thread real-time factors, and choosing 512 or 384 on the result. | 2–4 days |
| WebGPU | WGSL kernels for dilated Conv1d, transposed Conv1d and fused Snake (polynomial), plus f16 weights. Simpler than the EnCodec port: no LSTM, no GroupNorm. | 1–1.5 weeks |
| Encoding | DAC's encoder is about 22M params and 30.6 GMAC/s per channel. **Recommendation:** an offline Python encode tool now (a small `encode_dac.py` here writing codes, packed by picotone), 1–2 days. Port the encoder to Rust (the same conv/Snake kernels plus factorised l2-normalised RVQ) before release, so that `picotone` stays one command: about 1 week. Assets are encoded offline, so encoder speed doesn't matter. | 1–2 days, then about 1 week |
| New `.pico` scheme | A model enum for the DAC student plus its weight hash; 44.1 kHz; 86.13 frames/s; 10-bit codes; two mono streams; no scales and no segment stride. Pages are frame ranges; the decoder takes context from the neighbouring frames, so pages need a small decode overlap. Granule positions, pre-skip, loops and seeking follow FORMAT.md. | 3–5 days |
| cx2 table and net for DAC codes | Refit the tables and the net on DAC codes. **Gains will be small:** the study measured 9.41 bits per code zeroth-order for codebook 0 and about 9.94–9.99 for the rest (of 10), with repeat rates of 0.2–2%. Context modelling might win 3–8%. Do it only after measuring the gain with `tools\context_entropy.py`-style probes; until then, store codes raw (8.6 kbps at q5). | 1–2 weeks, maybe skipped |

## Smoke results (CPU, build machine, 6 threads, another test suite running)

Command: `uv run --extra cpu python smoke.py --manifest <pd.jsonl> --out <dir> --steps 240 --batch 2 --resume-steps 20 --timeout 1500`.
It took 21 min in all.

- **The MAC and parameter report prints.** Its numbers are the table at the top. Stock DAC measured
  69.26 GMAC/s and 54.09M parameters, matching the earlier study.
- **Bench** (CPU, 8,192-sample segments): batch 2 took 3.47 s per step, batch 4 took 5.51 s. That is a marginal
  1.0 s per item, which is about 75 GMAC/s.
- **Resume is bit-exact.** 20 steps straight, and 10 steps then `--resume` for 10 more, both end with weight
  hash `d11b113b1aa3dcc6`.
- **Losses fall.** 240 steps, batch 2, on three CC0/PD clips (first 30 s each), at about 3.3 s per step:

  | Loss | Start | After 240 steps |
  |---|---|---|
  | mel | 7.68 | 2.98 |
  | feature matching | 21.6 | 15.6 |
  | adversarial | 6.68 | 4.28 |
  | discriminator | 7.39 | 2.65 |

- **Held-out eval works.** Two picotone test clips, 3 s each: log-mel went 26.0 → 14.5 → 12.0 dB at steps
  0 → 120 → 240. Stock DAC q5 on the same audio scores 2.81 dB. Each eval wrote `listen/*__s<step>_q5.wav`
  next to `__0_original.wav` and `__0_stockDAC_q5.wav`, and printed the verdict (FAIL, as expected after 240
  steps).
- **The Jamendo-style fixture scans correctly.** It was a zip per artist with album folders, `license.txt`
  and a readme:
  - Codecs were detected from their bytes: Ogg Vorbis, Ogg FLAC (through ffmpeg) and Opus, plus a WAV named
    `.flac`.
  - Licences were read from `license.txt`: BY-SA and BY.
  - Artist and title came from the path.
  - **A 128 kbps MP3 transcoded to Ogg FLAC** (it reports 1.6 Mbps) was rejected on bandwidth (16.8 kHz), and
    so was the MP3 transcoded to Vorbis (17.3 kHz). The real tracks measured 20.4–21.9 kHz.
  - The loader read stored and deflated zip members, and its batches are deterministic.

## Licences

- DAC's code and weights are MIT (`LICENSES/DAC-MIT.txt`, Copyright (c) 2023-present, Descript).
  **A distilled student is a derivative of DAC's MIT weights;** ship that notice with it.
- The training audio keeps its own licences. CC-BY and BY-SA tracks need credit: `export.py --manifest` writes
  the credits list, with each track's licence URL, for the policy you chose.
- **BY-SA training data may make the trained weights a ShareAlike derivative (legally unsettled),** so the
  credits and the BY-SA notice (`.NOTICE.txt`) should ship with the exported student.
