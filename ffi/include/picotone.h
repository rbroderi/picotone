/* The C interface of picotone-ffi (ffi/src/lib.rs): decoding .pico files
 * (the foobar2000 component) and encoding them (the FFmpeg wrapper). All
 * strings are UTF-8. A failing call returns -1 (or null); pt_last_error()
 * then gives the message for this thread.
 *
 * LGPL-3.0-or-later (foobar/COPYING.LESSER). */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { PT_CODEC_HQ = 0, PT_CODEC_ENCODEC_48K = 1, PT_CODEC_ENCODEC_24K = 2 };

typedef struct PtHeader {
    uint32_t codec;          /* PT_CODEC_* */
    uint32_t sample_rate;    /* Hz */
    uint32_t channels;
    uint32_t codebooks;      /* HQ: of the first coded channel, with the layer when it fits */
    uint32_t side_codebooks; /* HQ mid/side: the side's; otherwise 0 */
    uint32_t layout;         /* HQ: 0 mono, 1 left/right, 2 mid/side, 3 parametric; EnCodec: 255 */
    uint32_t looped;
    uint32_t layered;        /* 1 when the given layer fits and is used */
    uint64_t length;         /* playable samples per channel */
} PtHeader;

typedef struct PtStream PtStream;
typedef struct PtEncoder PtEncoder;

/* Loads the cx2 tables from tables_dir; the networks (decoders and encoders)
 * load from weights_dir when first needed. Either may be null for the
 * command line's default: $PICOTONE_WEIGHTS / $PICOTONE_TABLES, then
 * weights/ / models/ next to the executable or its parents, then the
 * picotone project's. Only the first successful call counts. */
int32_t pt_init(const char *weights_dir, const char *tables_dir);
const char *pt_last_error(void);

/* Headers only: no network is loaded. layer may be null with layer_len 0. */
int32_t pt_header(const uint8_t *pico, size_t pico_len, const uint8_t *layer, size_t layer_len,
                  PtHeader *out);
/* Tags as "KEY=value\0...\0\0"; returns the bytes needed (written when they fit). */
ptrdiff_t pt_tags(const uint8_t *pico, size_t pico_len, char *buf, size_t cap);

/* Opens for decoding from playable sample start_sample. A layer that does not
 * fit fails the call (open again without it). The buffers may be freed after. */
PtStream *pt_open(const uint8_t *pico, size_t pico_len, const uint8_t *layer, size_t layer_len,
                  uint64_t start_sample);
/* 1: *data holds *frames interleaved frames (valid until the next call);
 * 0: the end; -1: failure. */
int32_t pt_decode(PtStream *stream, uint32_t max_units, const float **data, size_t *frames);
void pt_close(PtStream *stream);

/* A new encoder for interleaved 32-bit float input at sample_rate Hz with
 * channels channels (any rate and channel count the command line's audio
 * reader accepts; it is converted as the command line converts a file),
 * with `picotone encode`'s default options. Null on failure. One thread at
 * a time per encoder. */
PtEncoder *pt_encoder_new(uint32_t sample_rate, uint32_t channels);
/* Sets an option named as the `picotone encode` long flag without the
 * dashes, value as its text: "model" (hq|encodec-48k|encodec-24k), "fast"
 * ("1"/"0"), "channel-format" (auto|mono|stereo), "stereo"
 * (pick|ms|lr|parametric), "bandwidth", "preset" (music24|music12|music6),
 * "codebooks", "side-codebooks", "layer-codebooks", "layer-side-codebooks",
 * "ps-bands", "ps-frames", "stride", "crossfade" (triangle|edge), "beam",
 * "loop" ("1"/"0"). 0, or -1 for an unknown key or a value that does not
 * parse; options that do not go together fail pt_encoder_finish. */
int32_t pt_encoder_set(PtEncoder *enc, const char *key, const char *value);
/* Adds a "KEY=value" comment tag (as --tag). 0 or -1. */
int32_t pt_encoder_tag(PtEncoder *enc, const char *tag);
/* Appends frames interleaved frames (frames x channels floats). 0 or -1. */
int32_t pt_encoder_push(PtEncoder *enc, const float *data, size_t frames);
/* Encodes everything pushed, byte for byte as `picotone encode` would.
 * pico and pico_len get the .pico file, layer and layer_len the .picol layer
 * (null/0 when none; layer and layer_len may be null); valid until
 * pt_encoder_free. The encoder networks load on first use and are kept;
 * encodes run one at a time. Only once per encoder; pt_init must have
 * succeeded. 0 or -1. */
int32_t pt_encoder_finish(PtEncoder *enc, const uint8_t **pico, size_t *pico_len,
                          const uint8_t **layer, size_t *layer_len);
void pt_encoder_free(PtEncoder *enc); /* null is ignored */

#ifdef __cplusplus
}
#endif
