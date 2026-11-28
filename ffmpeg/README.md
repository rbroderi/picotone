# FFmpeg with picotone

`libpicotone` adds `.pico` reading and writing to FFmpeg 8.1 through picotone's C library
(`ffi/`, header `ffi/include/picotone.h`):

- **Demuxer `libpicotone`**: decodes a `.pico` file (HQ or EnCodec) to 32-bit float PCM, with its
  tags, duration and the `.picol` layer next to it. A seek reopens the file at the sample asked
  for and gives exactly the samples of a decode from the start.
- **Muxer `libpicotone`**: encodes any input FFmpeg can read into a `.pico` file (and its `.picol`
  layer), byte for byte as `picotone encode` codes the same samples with the same options.
  picotone's encoders code a whole signal at once, so the file is written when the input ends.

picotone decodes inside the demuxer, so FFmpeg sees the stream as `pcm_f32le`, not as a codec of
its own. Encodes are the same bytes on every platform. Decodes equal `picotone decode --float` bit for
bit except where HQ's q1–q5 post-filter runs; there the platform's maths library (glibc, MinGW,
MSVC) rounds its `exp` and `tanh` differently, by at most about 6e-6 (−104 dB).

## Licence

The FFmpeg files here (`libavformat/libpicotone*`) are LGPL-2.1-or-later, like FFmpeg's. picotone
is GPL-3.0-or-later, so `--enable-libpicotone` needs `--enable-gpl --enable-version3`, and the
programs built are GPLv3.

## Building

On Linux or WSL, with `build-essential nasm pkg-config` (and `mingw-w64` for Windows) and rustup's
stable toolchain (with the `x86_64-pc-windows-gnu` target for Windows):

```
bash ffmpeg/build.sh linux      # ffmpeg/dist/linux/ffmpeg, ffprobe
bash ffmpeg/build.sh windows    # ffmpeg/dist/windows/ffmpeg.exe, ffprobe.exe (MinGW-w64, static)
```

The script clones FFmpeg `n8.1` into `~/pt/ffmpeg` (or `$PICOTONE_FFMPEG_WORK`), adds picotone with
`apply.py`, builds the C library with cargo and writes a `picotone.pc` for FFmpeg's configure.
`apply.py` also works on its own (`python3 ffmpeg/apply.py path/to/ffmpeg`), for adding picotone
to another FFmpeg build. It registers the files in `configure`, `libavformat/Makefile` and
`allformats.c`, and makes the Ogg demuxer score a picotone file one lower so that `libpicotone`
reads it.

## Running

The model weights and cx2 tables are read at run time from `$PICOTONE_WEIGHTS` and
`$PICOTONE_TABLES`, or `weights/` and `models/` next to the program (the demuxer and muxer options
`weights` and `tables` name them per file).

```
ffmpeg -i song.pico song.flac
ffmpeg -i song.flac song.pico                                # HQ, 9 codebooks
ffmpeg -i song.flac -codebooks 3 -layer_codebooks 9 song.pico  # also writes song.picol
ffmpeg -i speech.wav -fast 1 speech.pico                     # EnCodec
ffmpeg -ss 300 -i song.pico -t 20 part.wav                   # exact seek
```

Muxer options, as `picotone encode`'s flags: `model`, `fast`, `channel_format`, `stereo`,
`bandwidth`, `preset`, `codebooks`, `side_codebooks`, `layer_codebooks`, `layer_side_codebooks`,
`ps_bands`, `ps_frames`, `stride`, `crossfade`, `beam`, `loop`. The input's metadata become
`KEY=value` tags. Demuxer options: `layer` (0 to ignore the `.picol`), `units` (about a second of
audio per packet, default 1).
