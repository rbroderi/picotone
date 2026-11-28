#!/usr/bin/env bash
# Builds FFmpeg with libpicotone (lean: FFmpeg's own codecs plus picotone).
#
#   ffmpeg/build.sh linux|windows [ffmpeg-source-dir]
#
# Run on Linux (or WSL). windows cross-compiles ffmpeg.exe with MinGW-w64.
# Needs: build-essential nasm pkg-config (and mingw-w64 for windows), and
# rustup with the stable toolchain (plus target x86_64-pc-windows-gnu).
# The result is GPLv3 (picotone is GPL-3.0-or-later). Output: ffmpeg/dist/<target>/.
set -euo pipefail

target=${1:?usage: build.sh linux|windows [ffmpeg-source-dir]}
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/.." && pwd)
work=${PICOTONE_FFMPEG_WORK:-$HOME/pt}
version=n8.1
src=${2:-$work/ffmpeg}
jobs=$(nproc)

case $target in
  linux)   rust_target=x86_64-unknown-linux-gnu
           cross=(--extra-ldflags=-static-libgcc) ;;
  # -static: no MinGW runtime DLLs next to ffmpeg.exe.
  windows) rust_target=x86_64-pc-windows-gnu
           cross=(--arch=x86_64 --target-os=mingw32 --cross-prefix=x86_64-w64-mingw32-
                  --extra-ldflags=-static) ;;
  *) echo "unknown target $target" >&2; exit 2 ;;
esac

if [ ! -d "$src" ]; then
  git clone --depth 1 --branch "$version" https://git.ffmpeg.org/ffmpeg.git "$src"
fi
python3 "$here/apply.py" "$src"

# libpicotone: the picotone-ffi static library, with a pkg-config file.
prefix=$work/prefix-$target
mkdir -p "$prefix/lib/pkgconfig" "$prefix/include"
export CARGO_TARGET_DIR=$work/cargo-target
# rustc prints the system libraries a static library needs only when it
# links, so the line is kept for builds that find nothing to do.
log=$prefix/cargo.log
rustup run stable cargo rustc --release --locked --manifest-path "$repo/ffi/Cargo.toml" \
  --target "$rust_target" --lib --crate-type staticlib -- --print native-static-libs 2>&1 | tee "$log"
line=$(sed -n 's/.*native-static-libs: //p' "$log" | tail -1)
if [ -n "$line" ]; then echo "$line" > "$prefix/native-libs.txt"; fi
native=$(cat "$prefix/native-libs.txt")
cp "$CARGO_TARGET_DIR/$rust_target/release/libpicotone_ffi.a" "$prefix/lib/"
cp "$repo/ffi/include/picotone.h" "$prefix/include/"
cat > "$prefix/lib/pkgconfig/picotone.pc" <<EOF
prefix=$prefix
Name: picotone
Description: picotone neural audio codec (C interface)
Version: 0.1.0
Libs: -L\${prefix}/lib -lpicotone_ffi $native
Cflags: -I\${prefix}/include
EOF

build=$work/build-$target
mkdir -p "$build"
cd "$build"
PKG_CONFIG_PATH=$prefix/lib/pkgconfig PKG_CONFIG_LIBDIR=$prefix/lib/pkgconfig \
"$src/configure" "${cross[@]}" \
  --pkg-config=pkg-config --pkg-config-flags=--static \
  --enable-gpl --enable-version3 --enable-libpicotone \
  --enable-static --disable-shared --disable-doc --disable-debug
make -j"$jobs"

dist=$here/dist/$target
mkdir -p "$dist"
case $target in
  linux)   cp ffmpeg ffprobe "$dist/" ;;
  windows) cp ffmpeg.exe ffprobe.exe "$dist/" ;;
esac
echo "built: $dist"
