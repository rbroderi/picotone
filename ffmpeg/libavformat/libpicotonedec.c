/*
 * .pico demuxer through libpicotone
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * picotone (.pico) files decoded by libpicotone into 32-bit float PCM.
 *
 * libpicotone's player decodes a whole file from any sample: a seek reopens
 * it there and gives exactly the samples of a decode from the start. The
 * file's .picol enhancement layer, when one sits next to it, is used too.
 */

#include <picotone.h>

#include "libavutil/avstring.h"
#include "libavutil/channel_layout.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "avformat.h"
#include "demux.h"
#include "internal.h"
#include "libpicotone.h"

typedef struct PicotoneDemuxContext {
    const AVClass *class;
    uint8_t *file;
    size_t file_size;
    uint8_t *layer;
    size_t layer_size;
    PtHeader header;
    PtStream *stream;
    /** Next sample to return, per channel. */
    int64_t position;

    /* options */
    char *weights;
    char *tables;
    int use_layer;
    int units;
} PicotoneDemuxContext;

#define OFFSET(x) offsetof(PicotoneDemuxContext, x)
#define D (AV_OPT_FLAG_AUDIO_PARAM | AV_OPT_FLAG_DECODING_PARAM)
static const AVOption options[] = {
    { "weights", "folder of the model weights (default: $PICOTONE_WEIGHTS, then weights/ next to the program)",
      OFFSET(weights), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, D },
    { "tables",  "folder of the cx2 tables (default: $PICOTONE_TABLES, then models/ next to the program)",
      OFFSET(tables),  AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, D },
    { "layer",   "use the .picol layer next to the file", OFFSET(use_layer), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, D },
    { "units",   "units (about a second each) decoded per packet", OFFSET(units), AV_OPT_TYPE_INT, { .i64 = 1 }, 1, 64, D },
    { NULL }
};

static int probe_picotone(const AVProbeData *p)
{
    const uint8_t *b = p->buf;
    int segments, offset;

    /* An Ogg page whose first packet is picotone's "Pico" identification header. */
    if (p->buf_size < 27 || memcmp(b, "OggS", 4) || !(b[5] & 0x02))
        return 0;
    segments = b[26];
    offset = 27 + segments;
    if (p->buf_size < offset + 4)
        return 0;
    if (memcmp(b + offset, "Pico", 4))
        return 0;
    return AVPROBE_SCORE_MAX;
}

static int open_stream(AVFormatContext *s, int64_t start)
{
    PicotoneDemuxContext *c = s->priv_data;

    pt_close(c->stream);
    c->stream = pt_open(c->file, c->file_size, c->layer, c->layer_size, start);
    if (!c->stream) {
        av_log(s, AV_LOG_ERROR, "picotone: %s\n", pt_last_error());
        return AVERROR_EXTERNAL;
    }
    c->position = start;
    return 0;
}

static void read_layer(AVFormatContext *s)
{
    PicotoneDemuxContext *c = s->priv_data;
    AVIOContext *pb = NULL;
    PtHeader with;
    char *url;
    size_t len = strlen(s->url);

    if (!c->use_layer || len < 5 || av_strcasecmp(s->url + len - 5, ".pico"))
        return;
    url = av_asprintf("%sl", s->url);
    if (!url)
        return;
    if (s->io_open(s, &pb, url, AVIO_FLAG_READ, NULL) >= 0) {
        if (ff_picotone_read_all(pb, &c->layer, &c->layer_size) < 0)
            c->layer_size = 0;
        ff_format_io_close(s, &pb);
    }
    if (c->layer_size &&
        (pt_header(c->file, c->file_size, c->layer, c->layer_size, &with) < 0 || !with.layered)) {
        av_log(s, AV_LOG_WARNING, "%s does not fit this file; decoding without it\n", url);
        av_freep(&c->layer);
        c->layer_size = 0;
    }
    if (c->layer_size)
        av_log(s, AV_LOG_VERBOSE, "using the layer %s\n", url);
    av_free(url);
}

static void read_tags(AVFormatContext *s)
{
    PicotoneDemuxContext *c = s->priv_data;
    ptrdiff_t need = pt_tags(c->file, c->file_size, NULL, 0);
    char *buf, *tag;

    if (need <= 1)
        return;
    buf = av_malloc(need);
    if (!buf)
        return;
    if (pt_tags(c->file, c->file_size, buf, need) == need) {
        for (tag = buf; *tag; tag += strlen(tag) + 1) {
            char *eq = strchr(tag, '=');
            if (eq) {
                *eq = 0;
                av_dict_set(&s->metadata, tag, eq + 1, AV_DICT_APPEND);
                *eq = '=';
            }
        }
    }
    av_free(buf);
}

static int read_header_picotone(AVFormatContext *s)
{
    PicotoneDemuxContext *c = s->priv_data;
    AVStream *st;
    int ret;

    if ((ret = ff_picotone_init(s, c->weights, c->tables)) < 0)
        return ret;
    if ((ret = ff_picotone_read_all(s->pb, &c->file, &c->file_size)) < 0)
        return ret;
    if (pt_header(c->file, c->file_size, NULL, 0, &c->header) < 0) {
        av_log(s, AV_LOG_ERROR, "picotone: %s\n", pt_last_error());
        return AVERROR_INVALIDDATA;
    }
    read_layer(s);
    if (c->layer_size)
        pt_header(c->file, c->file_size, c->layer, c->layer_size, &c->header);
    read_tags(s);

    st = avformat_new_stream(s, NULL);
    if (!st)
        return AVERROR(ENOMEM);
    st->codecpar->codec_type  = AVMEDIA_TYPE_AUDIO;
    st->codecpar->codec_id    = AV_NE(AV_CODEC_ID_PCM_F32BE, AV_CODEC_ID_PCM_F32LE);
    st->codecpar->sample_rate = c->header.sample_rate;
    st->codecpar->bits_per_coded_sample = 32;
    av_channel_layout_default(&st->codecpar->ch_layout, c->header.channels);
    avpriv_set_pts_info(st, 64, 1, c->header.sample_rate);
    st->start_time = 0;
    st->duration   = c->header.length;
    if (c->header.length)
        st->codecpar->bit_rate = av_rescale(c->file_size + c->layer_size, 8LL * c->header.sample_rate,
                                            c->header.length);

    return open_stream(s, 0);
}

static int read_packet_picotone(AVFormatContext *s, AVPacket *pkt)
{
    PicotoneDemuxContext *c = s->priv_data;
    const float *data;
    size_t frames = 0;
    int ret;

    do {
        ret = pt_decode(c->stream, c->units, &data, &frames);
        if (ret < 0) {
            av_log(s, AV_LOG_ERROR, "picotone: %s\n", pt_last_error());
            return AVERROR_EXTERNAL;
        }
        if (ret == 0)
            return AVERROR_EOF;
    } while (!frames);

    if ((ret = av_new_packet(pkt, frames * c->header.channels * sizeof(float))) < 0)
        return ret;
    memcpy(pkt->data, data, pkt->size);
    pkt->pts = pkt->dts = c->position;
    pkt->duration = frames;
    pkt->stream_index = 0;
    c->position += frames;
    return 0;
}

static int read_seek_picotone(AVFormatContext *s, int stream_index, int64_t ts, int flags)
{
    PicotoneDemuxContext *c = s->priv_data;

    if (stream_index < 0)
        ts = av_rescale(ts, c->header.sample_rate, AV_TIME_BASE);
    ts = av_clip64(ts, 0, c->header.length);
    return open_stream(s, ts);
}

static int read_close_picotone(AVFormatContext *s)
{
    PicotoneDemuxContext *c = s->priv_data;

    pt_close(c->stream);
    c->stream = NULL;
    av_freep(&c->file);
    av_freep(&c->layer);
    return 0;
}

static const AVClass picotone_demuxer_class = {
    .class_name = "libpicotone demuxer",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

const FFInputFormat ff_libpicotone_demuxer = {
    .p.name         = "libpicotone",
    .p.long_name    = NULL_IF_CONFIG_SMALL("picotone (libpicotone)"),
    .p.priv_class   = &picotone_demuxer_class,
    .p.extensions   = "pico",
    .priv_data_size = sizeof(PicotoneDemuxContext),
    .flags_internal = FF_INFMT_FLAG_INIT_CLEANUP,
    .read_probe     = probe_picotone,
    .read_header    = read_header_picotone,
    .read_packet    = read_packet_picotone,
    .read_seek      = read_seek_picotone,
    .read_close     = read_close_picotone,
};
