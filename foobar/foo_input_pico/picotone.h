/* The C interface of picotone-ffi (foobar/picotone-ffi/src/lib.rs). All
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

/* Loads the cx2 tables from tables_dir; the networks load from weights_dir
 * when a file first needs them. Only the first successful call counts. */
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

#ifdef __cplusplus
}
#endif
