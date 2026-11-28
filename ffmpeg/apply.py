"""Adds the libpicotone demuxer and muxer to an FFmpeg source tree.

    python apply.py path/to/ffmpeg

Copies libavformat/libpicotone*.{c,h} into the tree and registers them in
configure, libavformat/Makefile and libavformat/allformats.c, and makes the
Ogg demuxer step aside for picotone files when libpicotone is enabled. Safe
to run twice. Written against FFmpeg 8.1.
"""

import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
FILES = ["libpicotone.c", "libpicotone.h", "libpicotonedec.c", "libpicotoneenc.c"]


def edit(path: pathlib.Path, changes: list[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    for anchor, new in changes:
        if new in text:
            continue
        if text.count(anchor) != 1:
            sys.exit(f"{path}: expected one {anchor!r}")
        text = text.replace(anchor, new)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    tree = pathlib.Path(sys.argv[1]).resolve()
    lavf = tree / "libavformat"
    for name in FILES:
        shutil.copyfile(HERE / "libavformat" / name, lavf / name)

    edit(tree / "configure", [
        ("  --enable-libopenmpt      enable decoding tracked files via libopenmpt [no]\n",
         "  --enable-libopenmpt      enable decoding tracked files via libopenmpt [no]\n"
         "  --enable-libpicotone     enable .pico reading and writing via libpicotone [no]\n"),
        ('EXTERNAL_LIBRARY_GPLV3_LIST="\n    libsmbclient\n',
         'EXTERNAL_LIBRARY_GPLV3_LIST="\n    libpicotone\n    libsmbclient\n'),
        ('libopenmpt_demuxer_deps="libopenmpt"\n',
         'libopenmpt_demuxer_deps="libopenmpt"\n'
         'libpicotone_demuxer_deps="libpicotone"\n'
         'libpicotone_muxer_deps="libpicotone"\n'),
        ("enabled libopenmpt        && require_pkg_config",
         "enabled libpicotone       && require_pkg_config libpicotone picotone picotone.h pt_init\n"
         "enabled libopenmpt        && require_pkg_config"),
    ])
    edit(lavf / "Makefile", [
        ("OBJS-$(CONFIG_LIBOPENMPT_DEMUXER)        += libopenmpt.o\n",
         "OBJS-$(CONFIG_LIBOPENMPT_DEMUXER)        += libopenmpt.o\n"
         "OBJS-$(CONFIG_LIBPICOTONE_DEMUXER)       += libpicotonedec.o libpicotone.o\n"
         "OBJS-$(CONFIG_LIBPICOTONE_MUXER)         += libpicotoneenc.o libpicotone.o\n"),
    ])
    edit(lavf / "allformats.c", [
        ("extern const FFInputFormat  ff_libopenmpt_demuxer;\n",
         "extern const FFInputFormat  ff_libopenmpt_demuxer;\n"
         "extern const FFInputFormat  ff_libpicotone_demuxer;\n"
         "extern const FFOutputFormat ff_libpicotone_muxer;\n"),
    ])
    # A picotone file is Ogg whose first packet is its "Pico" header; the Ogg
    # demuxer cannot decode it, so it scores one less and libpicotone wins.
    edit(lavf / "oggdec.c", [
        ('#include <stdio.h>\n#include "libavutil/avassert.h"\n',
         '#include <stdio.h>\n#include "config_components.h"\n#include "libavutil/avassert.h"\n'),
        ('    if (!memcmp("OggS", p->buf, 5) && p->buf[5] <= 0x7)\n'
         "        return AVPROBE_SCORE_MAX;\n",
         '    if (!memcmp("OggS", p->buf, 5) && p->buf[5] <= 0x7) {\n'
         "#if CONFIG_LIBPICOTONE_DEMUXER\n"
         "        int offset = 27 + (p->buf_size > 26 ? p->buf[26] : 0);\n"
         '        if (p->buf_size >= offset + 4 && !memcmp(p->buf + offset, "Pico", 4))\n'
         "            return AVPROBE_SCORE_MAX - 1;\n"
         "#endif\n"
         "        return AVPROBE_SCORE_MAX;\n"
         "    }\n"),
    ])
    print(f"libpicotone added to {tree}")


if __name__ == "__main__":
    main()
