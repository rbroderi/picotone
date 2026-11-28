/*
 * Shared helpers of the libpicotone demuxer and muxer
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

#include <picotone.h>

#include "libavutil/bprint.h"
#include "libavutil/error.h"
#include "libavutil/log.h"
#include "libavutil/mem.h"
#include "libpicotone.h"

int ff_picotone_init(void *log_ctx, const char *weights, const char *tables)
{
    /* Only the first successful call loads anything; later ones return at once. */
    if (pt_init(weights, tables) < 0) {
        av_log(log_ctx, AV_LOG_ERROR, "picotone: %s\n", pt_last_error());
        return AVERROR_EXTERNAL;
    }
    return 0;
}

int ff_picotone_read_all(AVIOContext *pb, uint8_t **out, size_t *out_size)
{
    AVBPrint bp;
    char *data;
    int ret;

    av_bprint_init(&bp, 0, AV_BPRINT_SIZE_UNLIMITED);
    ret = avio_read_to_bprint(pb, &bp, SIZE_MAX);
    if (ret < 0) {
        av_bprint_finalize(&bp, NULL);
        return ret;
    }
    if (!av_bprint_is_complete(&bp)) {
        av_bprint_finalize(&bp, NULL);
        return AVERROR(ENOMEM);
    }
    *out_size = bp.len;
    ret = av_bprint_finalize(&bp, &data);
    if (ret < 0)
        return ret;
    *out = (uint8_t *)data;
    return 0;
}
