# Sample song

**Ravel – Boléro**, by Jason Weinberger ([SoundCloud](https://soundcloud.com/jasonweinberger/ravel-bolero)),
licensed [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). About 16 minutes, 44.1 kHz stereo.
It is the reference piece for listening to picotone's settings: it starts as a quiet snare and flute and ends with
the full orchestra.

| File | What it is |
|---|---|
| `ravel_bolero.flac` | The original, 24-bit FLAC (Git LFS) |
| `ravel_bolero_hq_q3.pico`, `_q9`, `_q16` | HQ at 3, 9 and 16 codebooks per channel |
| `ravel_bolero_hq_q3_layered.pico` + `.picol` | HQ q3, with a `.picol` layer that lifts it to q16 when it lies next to the file |
| `ravel_bolero_fast_q3.pico`, `_q9`, `_q16` | EnCodec 48 kHz (`--fast`) at 3, 9 and 16 codebooks |

Each file's title gives its setting, such as `Boléro (HQ q9, 13 kbps)`, so players list them apart.

The `.pico` and `.picol` files are adaptations of the recording and are licensed CC BY-SA 3.0 as well, credited to
Jason Weinberger (see `ravel_bolero.LICENSE.txt`); their tags carry the credit and licence. They are re-encoded
whenever a table or the format changes. To make them again:

```powershell
picotone encode samples\ravel_bolero.flac samples\ravel_bolero_hq_q9.pico --codebooks 9 --tag "TITLE=Boléro (HQ q9, 13 kbps)"
picotone encode samples\ravel_bolero.flac samples\ravel_bolero_hq_q3_layered.pico --codebooks 3 --layer-codebooks 16
picotone encode samples\ravel_bolero.flac samples\ravel_bolero_fast_q9.pico --fast --codebooks 9
```
