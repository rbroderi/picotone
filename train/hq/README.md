# train/hq: training the HQ networks

This folder trains the three networks of picotone's HQ codec and exports them for the Rust code:

| Network | Shipped as | What it is |
|---|---|---|
| STFT encoder | `weights/dac44_stftenc` | maps 44.1 kHz audio to DAC 44 kHz's latent (`encoder.py`, `STFTEncoder`) |
| Codebooks 10–16 | `weights/dac44_quantizer` (stages 9–15) | 7 codebooks after DAC's 9, trained with the decoder (`encoder.py`, `ExtRVQ`) |
| Vocos decoder | `weights/dac44_vocos` | decodes 1 to 16 codebooks (`vocos_student.py`, `make_decoder`) |
| Low-tier post-filter | `pf.*` in `weights/dac44_vocos` | a per-bin gain on the decoder's spectrum at q1–q5 (`pf_lowtier.py`, `LowTierPF`) |

DAC's own quantiser (codebooks 1–9) and DAC's encoder and decoder are used as they are, from
descript-audio-codec 1.0.0's 44 kHz 8 kbps weights (MIT; `teacher.py` downloads them through
`dac.utils.download` on first use). The HQ cx2 table is trained separately, by `../hq_net.py` (see
[`../README.md`](../README.md)).

Everything is one training script, `train.py`, with `--arch vocos` (the decoder alone) or `--arch encoder`
(the encoder, and with `--enc-train-dec` the decoder, the extra codebooks and the post-filter behind it). A run
lives in `runs/<name>/`: checkpoints `ckpt_<step>.pt` (7 digits) and `ckpt_last.pt`, `evals.jsonl`, and
`listen/` WAVs. `--resume` continues a run from `ckpt_last.pt`; `--stop-at N` stops cleanly after step N's
checkpoint while the learning-rate schedule still spans `--steps`.

## Setup

```bash
cd train/hq
uv sync --extra cuda      # or --extra cpu for smoke tests
```

Commands below run from `train/hq/` and start with `uv run --extra cuda python`. Training needs a CUDA GPU; bf16 autocast is on by default on CUDA (`--amp off` if a loss goes
non-finite). Give every long run a `timeout` and check its log every 10–15 minutes; a run past three times its
expected time, or whose log has stopped, is hung.

## 1. Data

Training reads JSON-lines manifests in `data/`, one row per track: path, source, licence class and URL,
attribution, measured bandwidth, `keep` and `split`. Training uses only `split=train, keep=true` rows of the
licence classes `--licences` admits.

| Manifest | Made by | Audio |
|---|---|---|
| `data/pd.jsonl` | `prep_pd.py --pd-root <dir> --out data/pd.jsonl` | a folder of clips you provide from the public-domain, CC0 and CC BY sources in `../../DATA_SOURCES.json`: 48 kHz stereo WAVs in `<dir>/work/wav*/`, their clip lists in `<dir>/work/clips*.json` and the download manifests (licence and creator) in `<dir>/dl/manifest*.json` |
| `data/jamendo.jsonl` | `prep_corpus.py --root <dir> --source jamendo --out data/jamendo.jsonl --url-base https://archive.org/download/jamendo_complete_zip_2012_05_20` | archive.org's `jamendo_complete_zip_2012_05_20`, the zips as they are |
| `data/speech.jsonl`, `data/hits.jsonl` | `prep_extra.py --root <dir>` | VCTK 0.92 speech joined into 60 s files (CC BY 4.0), and drum and piano performances rendered from sfzinstruments sample libraries (CC0, CC BY) |
| `data/lv45/librivox.jsonl` | `prep_extra.py --root <dir> --stages librivox,manifests --sources librivox --data-dir data/lv45` | LibriVox readings (public domain) |

- **Quality filter:** `prep_corpus.py` measures each track's effective bandwidth on three windows and rejects
  tracks under `--min-bandwidth` (18000 Hz), which catches MP3 transcodes whatever their bitrate. The speech,
  hits and LibriVox sources are lossless and skip it.
- **Held out:** a fixed 2% of Jamendo artists, the picotone test clips (`data/picotone_testclips.json`) and every
  clip by a creator who appears in them. Each prep script also writes `<manifest>.eval.json`, fixed held-out
  excerpts used by `--eval-set`.
- **Licences:** `train.py --licences` takes a preset or a comma list of classes. The default, `sharealike`,
  admits `pd,cc0,by,by-sa,fal`; `strict` is `pd,cc0` and `permissive` adds `by`. NC licences are never admitted.
  Every admitted track is credited in the exported weights' `.credits.jsonl`, and the export's `.NOTICE.txt`
  carries the ShareAlike note when BY-SA or FAL audio was used.
- **Paths:** `--path-map OLD=NEW` (repeatable) rewrites the manifests' paths when the audio has moved to
  another machine or drive.
- **Re-runs** of the prep scripts skip work already done.

The shared data arguments of every run below:

```bash
DATA=(--manifest data/pd.jsonl --manifest data/jamendo.jsonl --manifest data/speech.jsonl
  --manifest data/hits.jsonl --manifest data/lv45/librivox.jsonl
  --mix jamendo=0.45,pd=0.2,librivox=0.15,speech=0.05,hits=0.15
  --eval-set data/pd.jsonl.eval.json --eval-set data/speech.jsonl.eval.json
  --eval-set data/lv45/librivox.jsonl.eval.json --eval-kinds speech
  --workers 8)
```

## 2. The Vocos decoder, on DAC's codes

The decoder learns to decode DAC's own codes (DAC's encoder and quantiser make them on the fly), with DAC's
decoder as the teacher: mel 15, multi-period and multi-resolution discriminators (adversarial 1, feature
matching 2), and a mel term against the teacher's output.

```bash
timeout 60h uv run --extra cuda python train.py --run runs/vocos_m512 --arch vocos --vocos-dim 512 \
  --vocos-blocks 8 --vocos-hop 512 --manifest data/pd.jsonl --manifest data/jamendo.jsonl --mix pd=0.4,jamendo=0.6 \
  --eval-set data/pd.jsonl.eval.json \
  --n-q 5 --eval-n-q 4,5,6,7,9 --batch 40 --w-teacher-mel 15 --teacher-amp fp32 --steps 80000 \
  --eval-every 5000 --listen-every 20000 > runs/vocos_m512.log 2>&1
```

Then a fine-tune on all sources at the tiers 4–9, adding a multi-resolution compressed complex STFT loss against
the teacher (`--w-teacher-cplx`; `--cplx-windows`, `--cplx-power` and `--cplx-alpha` shape it):

```bash
timeout 12h uv run --extra cuda python train.py --run runs/vocos_ft_cplx --arch vocos --vocos-dim 512 \
  --vocos-blocks 8 --vocos-hop 512 --init-from runs/vocos_m512/ckpt_last.pt "${DATA[@]}" \
  --n-q 4,5,6,7,9 --eval-n-q 4,5,6,7,9 --batch 40 --w-teacher-mel 15 --w-teacher-cplx 200 --teacher-amp fp32 \
  --lr 3e-5 --lr-schedule linear --steps 25000 --eval-every 2500 --keep-every 5000 > runs/vocos_ft_cplx.log 2>&1
```

## 3. The STFT encoder

The encoder (`--enc-arch stft`: STFT inputs at n_fft 2048 and 512 compressed by `|S|^0.3`, a k1 embed, 8
ConvNeXt blocks 512 wide, a Linear to DAC's 1024-dim latent) trains in front of DAC's frozen quantiser and the
frozen decoder from step 2:

```bash
NQ=5,5,5,5,5,5,5,5,5,5,5,5,5,9,9,9,9,9,9,9,9,9,4,4,4,4,4,4,4,6,6,6,6,6,7,7,7,7,7,1,1,1,2,2,2,2,3,3,3,3
E=(--arch encoder --enc-dec runs/vocos_ft_cplx/ckpt_last.pt --enc-ctx 12288 "${DATA[@]}")
```

`--n-q` lists tiers to draw from per item, so `NQ` weights them. **Phase 1, matching:** the encoder learns
DAC's pre-quantisation latent (an L1 term plus a cross-entropy on the codes), with no GAN:

```bash
timeout 8h uv run --extra cuda python train.py --run runs/encA_stft "${E[@]}" --enc-phase match \
  --batch 64 --lr 3e-4 --betas 0.9 0.99 --lr-schedule linear --steps 30000 --eval-every 5000 \
  --eval-n-q 5,9 > runs/encA_stft.log 2>&1
```

**Phase 2, end to end:** through the quantiser and the frozen decoder, on the decoder's spectral losses plus the
latent and commitment terms, with no GAN. At q1–q3 the items are scored on the band under 8 kHz
(`--bwe-cutoff 8000`). The run is on a 60k linear schedule and the encoder ships from its 20k EMA:

```bash
timeout 30h uv run --extra cuda python train.py --run runs/encB_arm4 "${E[@]}" --enc-phase e2e \
  --init-from runs/encA_stft/ckpt_last.pt --init-d runs/vocos_ft_cplx/ckpt_last.pt --batch 40 \
  --lr 1e-4 --lr-schedule linear --steps 60000 --stop-at 20000 --n-q "$NQ" --bwe-cutoff 8000 \
  --enc-w-z 1 --enc-w-commit 1 --w-adv 0 --w-fm 0 --eval-every 5000 --keep-every 10000 --eval-n-q 2,5,9 \
  > runs/encB_arm4.log 2>&1
```

## 4. The decoder on the encoder's codes

With the encoder frozen (`--enc-freeze-enc`), the decoder trains on a 50/50 mix of items coded by DAC's encoder
(tiers 4–9, with the teacher terms) and by the STFT encoder (tiers 1–9; q1–q3 scored under 8 kHz plus 0.3 of
the full-band mel term):

```bash
NQ3=4,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,7,7,7,7,7,7,9,9,9,9,9,9,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3
DEC=(--enc-phase e2e --enc-train-dec --enc-freeze-enc --enc-mix-dac 0.5 --n-q-dac 4,5,6,7,9 --bwe-cutoff 8000
  --bwe-hf-weight 0.3 --enc-w-z 0 --enc-w-commit 0 --enc-w-cplx 0 --w-mel 15 --w-teacher-mel 15
  --w-teacher-cplx 100 --teacher-amp fp32 --w-adv 1 --w-fm 2 --w-lf 100 --eval-n-q 2,5,9)
timeout 6h uv run --extra cuda python train.py --run runs/lowtier_ctrl5 "${E[@]}" "${DEC[@]}" --n-q "$NQ3" \
  --init-from runs/encB_arm4/ckpt_0020000.pt --init-from-ema --init-opt-from runs/vocos_ft_cplx/ckpt_last.pt \
  --init-d runs/vocos_ft_cplx/ckpt_last.pt --batch 40 --lr 3e-6 --lr-schedule warmconst --lr-warmup 500 \
  --steps 5000 --eval-every 5000 --keep-every 5000 > runs/lowtier_ctrl5.log 2>&1
```

## 5. Codebooks 10–16

`build_ext_cb.py` builds the extra codebooks' starting point in DAC's style, one after another on the residual
left after the codebooks before them (in-projection from the residual's top 8 PCA directions, 1024 entries by
spherical k-means), over training crops coded by the encoder (`--enc`, e.g.
`runs/encB_arm4/ckpt_0020000.pt`). `--base-ckpt` first applies a checkpoint's trained extra codebooks and
builds only the rest after them.

```bash
timeout 2h uv run --extra cuda python build_ext_cb.py --enc runs/encB_arm4/ckpt_0020000.pt --n-ext 7 --out runs/q16_ext/ext_cb_init.pt
```

Then the decoder (with its discriminator) and codebooks 10–16 train together from step 4's full state; the
codebooks learn from the output losses only (`--enc-ext-grad out`). Half the items are DAC-coded (tiers 4–9),
half STFT-encoder-coded: 60% at `NQ3`'s tiers, 40% spread evenly over q10–q16.

```bash
NQ16=$(python3 -c "a='$NQ3'.split(','); print(','.join(a*7 + [str(n) for n in range(10,17) for _ in range(30)]))")
C5=runs/lowtier_ctrl5/ckpt_0005000.pt
timeout 18h uv run --extra cuda python train.py --run runs/q16 "${E[@]}" "${DEC[@]}" --n-q "$NQ16" \
  --init-from "$C5" --init-opt-from "$C5" --enc-ext-cb runs/q16_ext/ext_cb_init.pt --enc-ext-lr 1e-4 \
  --enc-ext-grad out --enc-ext-w-commit 0 --enc-ext-w-codebook 0 --lr 1e-5 --lr-schedule warmlin \
  --lr-warmup 1000 --lr-d 3e-6 --lr-schedule-d warmconst --lr-warmup-d 500 --batch 40 --steps 30000 \
  --eval-every 5000 --keep-every 5000 > runs/q16.log 2>&1
```

The shipped decoder and codebooks are `runs/q16/ckpt_0030000.pt`'s EMA weights.

## 6. The low-tier post-filter

The post-filter trains after the decoder with everything else frozen (`--pf-lowtier`): the decoder, the encoder
and codebooks 1–16 keep their weights, and only the post-filter and a fresh discriminator learn. It reads the
decoder head's spectrum, the decoder's last hidden state and the tier, and with `--pflt-mode mag` gives a real
gain `exp(tanh(o))` per bin (`--pflt-mag-bound 1.0`), phase unchanged. Tiers q1–q5 only, weighted q1 20%,
q2 25%, q3 25%, q4 20%, q5 10%; at q6 and above it does not run. Losses: mel 15, adversarial 1, feature
matching 2, and a short-window log-magnitude loss (n_fft 128, 256, 512; `--w-short`).

```bash
NQPF=1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,5,5
timeout 20h uv run --extra cuda python train.py --run runs/pf_run_g "${E[@]}" --enc-phase e2e --enc-train-dec \
  --enc-freeze-enc --init-from runs/q16/ckpt_0030000.pt --init-from-ema --enc-ext-cb runs/q16_ext/ext_cb_init.pt \
  --enc-ext-w-commit 0 --enc-ext-w-codebook 0 --pf-lowtier --pflt-dim 192 --pflt-blocks 4 --pflt-max-q 5 \
  --pflt-mode mag --pflt-mag-bound 1.0 --n-q "$NQPF" --bwe-cutoff 8000 --bwe-hf-weight 0.3 \
  --lr 5e-5 --lr-schedule warmcos --lr-warmup 500 --fresh-d --gan-start 0 --gan-g-start 500 \
  --lr-d 2e-4 --lr-schedule-d warmconst --lr-warmup-d 500 --g-ema 0.999 --eval-n-q 3,5 --enc-w-z 0 \
  --enc-w-commit 0 --w-mel 15 --w-teacher-mel 0 --w-teacher-cplx 0 --enc-w-cplx 0 --w-adv 1 --w-fm 2 \
  --w-lf 100 --w-consist 0 --w-short 1.0 --batch 48 --steps 20000 --eval-every 2500 --keep-every 2500 \
  --ckpt-every 500 > runs/pf_run_g.log 2>&1
```

The shipped post-filter is `runs/pf_run_g/ckpt_0007500.pt`'s EMA weights. The owner picks the step by ear
from the listening renders.

## 7. Export to `weights/`

The exports only read checkpoints; `export_dec_ft.py`, `export_q16_codebooks.py` and `pf_export.py` refuse to
overwrite. First into PyTorch files here:

```bash
# Decoder: export.py exports a plain --arch vocos run (state dict, .json, .credits.jsonl, .NOTICE.txt) and
# makes the reference that export_dec_ft.py needs: --ref is the export of the decoder the fine-tune started
# from (checked tensor for tensor against --ref-ckpt). export_dec_ft.py adds a .PROVENANCE.txt.
uv run --extra cuda python export.py --ckpt runs/vocos_ft_cplx/ckpt_last.pt --out export/dac44_vocos \
  --manifest data/pd.jsonl --manifest data/jamendo.jsonl --manifest data/speech.jsonl \
  --manifest data/hits.jsonl --manifest data/lv45/librivox.jsonl
uv run --extra cuda python export_dec_ft.py --ckpt runs/q16/ckpt_0030000.pt --ema --ref export/dac44_vocos \
  --out export/dac44_vocos_q16
# Codebooks 10-16, in DAC's key names (quantizer.quantizers.9..15):
uv run --extra cuda python export_q16_codebooks.py --ckpt runs/q16/ckpt_0030000.pt --out export/q16_codebooks.pt
# Post-filter (runs/pf_run_g/ckpt_0007500.pt, EMA): pf_g7500.safetensors, its .json and test vectors
uv run --extra cuda python pf_export.py --ckpt runs/pf_run_g/ckpt_0007500.pt \
  --dec export/dac44_vocos_q16.pt --enc runs/encB_arm4/ckpt_0020000.pt --out export/pf
```

Then into picotone's Rust format with the scripts in `../../tools/` (run from here, in this environment):

| Script | Reads | Writes |
|---|---|---|
| `tools/export_stftenc.py --ckpt runs/encB_arm4/ckpt_0020000.pt --out ../../weights/dac44_stftenc` | the encoder's `G_ema` `enc.*` | `dac44_stftenc.safetensors`, `.json` |
| `tools/export_quantizer.py --dac <DAC weights .pth> --ext export/q16_codebooks.pt --out ../../weights/dac44_quantizer` | DAC's 9 codebooks and the 7 extra | `dac44_quantizer.safetensors`, `.json` |
| `tools/export_vocos.py --vocos export/dac44_vocos_q16.pt --dac <DAC weights .pth> --ext export/q16_codebooks.pt --out ../../weights/dac44_vocos` | the decoder and the 16 codebooks (folded into its embed conv) | `dac44_vocos.safetensors`, `.json`, with the `.NOTICE.txt` and `.credits.jsonl` |
| `tools/export_postfilter.py --pf export/pf/pf_g7500.safetensors --vocos ../../weights/dac44_vocos` | the post-filter | `pf.*` added to `dac44_vocos.safetensors` and `.json` |

`<DAC weights .pth>` is the `weights_44khz_8kbps_0.0.1.pth` that `dac.utils.download` fetched. The NOTICE and
credits files of `dac44_stftenc` and `dac44_quantizer` are written by hand from the decoder's
(`weights/*.NOTICE.txt`).

Check the Rust ports against PyTorch before shipping:

- `tools/stftenc_parity.py` (`prepare`, then `cargo test --release --test hq_stftenc -- --ignored
  real_weights_parity_files`, then `compare`): the encoder's latent SNR and DAC codes on real audio.
- `tools/stftenc_fixture.py`, `tools/vocos_fixture.py`, `tools/postfilter_fixture.py`: the small random-weight
  fixtures the `cargo test` suites decode.

New weights change the codes, so re-dump the codes and retrain the HQ cx2 table (`../README.md`), and re-encode
every `.pico` file.
