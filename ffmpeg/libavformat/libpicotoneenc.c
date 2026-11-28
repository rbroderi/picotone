/*
 * .pico muxer through libpicotone
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
 * picotone (.pico) files encoded by libpicotone from 32-bit float PCM.
 *
 * picotone's encoders code a whole signal at once, so the muxer collects
 * the samples and writes the file when the input ends. With layer_codebooks
 * set it also writes the .picol enhancement layer next to the output.
 */

#include <picotone.h>

#include "libavutil/avstring.h"
#include "libavutil/dict.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "avformat.h"
#include "internal.h"
#include "mux.h"
#include "libpicotone.h"

typedef struct PicotoneMuxContext {
    const AVClass *class;
    PtEncoder *encoder;

    /* options */
    char *weights;
    char *tables;
    char *model;
    int fast;
    char *channel_format;
    char *stereo;
    double bandwidth;
    char *preset;
    int codebooks;
    int side_codebooks;
    int layer_codebooks;
    int layer_side_codebooks;
    int ps_bands;
    int ps_frames;
    int stride;
    char *crossfade;
    int beam;
    int loop;
} PicotoneMuxContext;

#define OFFSET(x) offsetof(PicotoneMuxContext, x)
#define E (AV_OPT_FLAG_AUDIO_PARAM | AV_OPT_FLAG_ENCODING_PARAM)
static const AVOption options[] = {
    { "weights", "folder of the model weights (default: $PICOTONE_WEIGHTS, then weights/ next to the program)",
      OFFSET(weights), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, E },
    { "tables", "folder of the cx2 tables (default: $PICOTONE_TABLES, then models/ next to the program)",
      OFFSET(tables), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, E },
    { "model", "hq (default), encodec-48k or encodec-24k", OFFSET(model), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, E },
    { "fast", "EnCodec instead of HQ: encodec-24k for mono, encodec-48k for stereo",
      OFFSET(fast), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, E },
    { "channel_format", "auto (default), mono or stereo", OFFSET(channel_format), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, E },
    { "stereo", "HQ stereo layout: pick (default), ms, lr or parametric", OFFSET(stereo), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, E },
    { "bandwidth", "EnCodec bandwidth in kbps (default 6)", OFFSET(bandwidth), AV_OPT_TYPE_DOUBLE, { .dbl = 0 }, 0, 1000, E },
    { "preset", "EnCodec 48 kHz music preset: music24, music12 or music6", OFFSET(preset), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, E },
    { "codebooks", "codebooks per frame (HQ: 1 to 16 per channel, default 9)", OFFSET(codebooks), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 64, E },
    { "side_codebooks", "HQ mid/side: codebooks of the side", OFFSET(side_codebooks), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 64, E },
    { "layer_codebooks", "HQ: also write a .picol layer up to this many codebooks", OFFSET(layer_codebooks), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 64, E },
    { "layer_side_codebooks", "HQ layer: the side's codebooks", OFFSET(layer_side_codebooks), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 64, E },
    { "ps_bands", "HQ parametric stereo: parameter bands, 20 or 10", OFFSET(ps_bands), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 64, E },
    { "ps_frames", "HQ parametric stereo: frames per parameter set, 1, 2, 4, 8 or 16", OFFSET(ps_frames), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 64, E },
    { "stride", "EnCodec segment stride in samples", OFFSET(stride), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, INT_MAX, E },
    { "crossfade", "EnCodec cross-fade: triangle or edge", OFFSET(crossfade), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, E },
    { "beam", "EnCodec quantiser beam width (default 8)", OFFSET(beam), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 1024, E },
    { "loop", "code the input as a seamless loop", OFFSET(loop), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, E },
    { NULL }
};

static int set(AVFormatContext *s, const char *key, const char *value)
{
    PicotoneMuxContext *c = s->priv_data;

    if (pt_encoder_set(c->encoder, key, value) < 0) {
        av_log(s, AV_LOG_ERROR, "picotone %s=%s: %s\n", key, value, pt_last_error());
        return AVERROR(EINVAL);
    }
    return 0;
}

static int set_int(AVFormatContext *s, const char *key, int value)
{
    char text[32];

    if (!value)
        return 0;
    snprintf(text, sizeof(text), "%d", value);
    return set(s, key, text);
}

static int set_text(AVFormatContext *s, const char *key, const char *value)
{
    return value && *value ? set(s, key, value) : 0;
}

static int write_header_picotone(AVFormatContext *s)
{
    PicotoneMuxContext *c = s->priv_data;
    AVCodecParameters *par = s->streams[0]->codecpar;
    const AVDictionaryEntry *tag = NULL;
    int ret;

    if (par->codec_id != AV_NE(AV_CODEC_ID_PCM_F32BE, AV_CODEC_ID_PCM_F32LE)) {
        av_log(s, AV_LOG_ERROR, "the picotone muxer takes native 32-bit float PCM\n");
        return AVERROR(EINVAL);
    }
    if ((ret = ff_picotone_init(s, c->weights, c->tables)) < 0)
        return ret;
    c->encoder = pt_encoder_new(par->sample_rate, par->ch_layout.nb_channels);
    if (!c->encoder) {
        av_log(s, AV_LOG_ERROR, "picotone: %s\n", pt_last_error());
        return AVERROR_EXTERNAL;
    }

    if ((ret = set_text(s, "model", c->model)) < 0 ||
        (ret = (c->fast ? set(s, "fast", "1") : 0)) < 0 ||
        (ret = set_text(s, "channel-format", c->channel_format)) < 0 ||
        (ret = set_text(s, "stereo", c->stereo)) < 0 ||
        (ret = set_text(s, "preset", c->preset)) < 0 ||
        (ret = set_int(s, "codebooks", c->codebooks)) < 0 ||
        (ret = set_int(s, "side-codebooks", c->side_codebooks)) < 0 ||
        (ret = set_int(s, "layer-codebooks", c->layer_codebooks)) < 0 ||
        (ret = set_int(s, "layer-side-codebooks", c->layer_side_codebooks)) < 0 ||
        (ret = set_int(s, "ps-bands", c->ps_bands)) < 0 ||
        (ret = set_int(s, "ps-frames", c->ps_frames)) < 0 ||
        (ret = set_int(s, "stride", c->stride)) < 0 ||
        (ret = set_text(s, "crossfade", c->crossfade)) < 0 ||
        (ret = set_int(s, "beam", c->beam)) < 0 ||
        (ret = (c->loop ? set(s, "loop", "1") : 0)) < 0)
        return ret;
    if (c->bandwidth > 0) {
        char text[32];
        snprintf(text, sizeof(text), "%g", c->bandwidth);
        if ((ret = set(s, "bandwidth", text)) < 0)
            return ret;
    }

    /* The file's metadata become Vorbis-style KEY=value comments. */
    while ((tag = av_dict_iterate(s->metadata, tag))) {
        char *text, *p;
        if (!av_strcasecmp(tag->key, "encoder"))
            continue;
        text = av_asprintf("%s=%s", tag->key, tag->value);
        if (!text)
            return AVERROR(ENOMEM);
        for (p = text; *p != '='; p++)
            *p = av_toupper(*p);
        ret = pt_encoder_tag(c->encoder, text);
        av_free(text);
        if (ret < 0) {
            av_log(s, AV_LOG_WARNING, "tag %s left out: %s\n", tag->key, pt_last_error());
        }
    }
    return 0;
}

static int write_packet_picotone(AVFormatContext *s, AVPacket *pkt)
{
    PicotoneMuxContext *c = s->priv_data;
    int channels = s->streams[0]->codecpar->ch_layout.nb_channels;
    size_t frames = pkt->size / (channels * sizeof(float));

    if (pt_encoder_push(c->encoder, (const float *)pkt->data, frames) < 0) {
        av_log(s, AV_LOG_ERROR, "picotone: %s\n", pt_last_error());
        return AVERROR_EXTERNAL;
    }
    return 0;
}

static int write_layer(AVFormatContext *s, const uint8_t *layer, size_t size)
{
    AVIOContext *pb = NULL;
    char *url;
    size_t len = strlen(s->url);
    int ret;

    if (len < 5 || av_strcasecmp(s->url + len - 5, ".pico")) {
        av_log(s, AV_LOG_ERROR, "the .picol layer needs an output named *.pico\n");
        return AVERROR(EINVAL);
    }
    url = av_asprintf("%sl", s->url);
    if (!url)
        return AVERROR(ENOMEM);
    ret = s->io_open(s, &pb, url, AVIO_FLAG_WRITE, NULL);
    if (ret < 0) {
        av_log(s, AV_LOG_ERROR, "cannot write %s\n", url);
        av_free(url);
        return ret;
    }
    avio_write(pb, layer, size);
    av_log(s, AV_LOG_INFO, "wrote the layer %s (%zu bytes)\n", url, size);
    av_free(url);
    return ff_format_io_close(s, &pb);
}

static int write_trailer_picotone(AVFormatContext *s)
{
    PicotoneMuxContext *c = s->priv_data;
    const uint8_t *pico, *layer;
    size_t pico_size, layer_size;

    if (pt_encoder_finish(c->encoder, &pico, &pico_size, &layer, &layer_size) < 0) {
        av_log(s, AV_LOG_ERROR, "picotone: %s\n", pt_last_error());
        return AVERROR_EXTERNAL;
    }
    avio_write(s->pb, pico, pico_size);
    if (layer_size)
        return write_layer(s, layer, layer_size);
    return 0;
}

static void deinit_picotone(AVFormatContext *s)
{
    PicotoneMuxContext *c = s->priv_data;

    pt_encoder_free(c->encoder);
    c->encoder = NULL;
}

static const AVClass picotone_muxer_class = {
    .class_name = "libpicotone muxer",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

const FFOutputFormat ff_libpicotone_muxer = {
    .p.name           = "libpicotone",
    .p.long_name      = NULL_IF_CONFIG_SMALL("picotone (libpicotone)"),
    .p.extensions     = "pico",
    .p.mime_type      = "audio/ogg",
    .p.priv_class     = &picotone_muxer_class,
    .priv_data_size   = sizeof(PicotoneMuxContext),
    .p.audio_codec    = AV_NE(AV_CODEC_ID_PCM_F32BE, AV_CODEC_ID_PCM_F32LE),
    .p.video_codec    = AV_CODEC_ID_NONE,
    .p.subtitle_codec = AV_CODEC_ID_NONE,
    .p.flags          = AVFMT_NOTIMESTAMPS,
    .flags_internal   = FF_OFMT_FLAG_MAX_ONE_OF_EACH | FF_OFMT_FLAG_ONLY_DEFAULT_CODECS,
    .write_header     = write_header_picotone,
    .write_packet     = write_packet_picotone,
    .write_trailer    = write_trailer_picotone,
    .deinit           = deinit_picotone,
};
