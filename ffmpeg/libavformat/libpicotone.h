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

#ifndef AVFORMAT_LIBPICOTONE_H
#define AVFORMAT_LIBPICOTONE_H

#include <stddef.h>
#include <stdint.h>

#include "avio.h"

/**
 * Loads libpicotone's cx2 tables once per process. weights and tables may
 * be NULL for libpicotone's defaults.
 */
int ff_picotone_init(void *log_ctx, const char *weights, const char *tables);

/** Reads the rest of pb into a new buffer (freed with av_free). */
int ff_picotone_read_all(AVIOContext *pb, uint8_t **out, size_t *out_size);

#endif /* AVFORMAT_LIBPICOTONE_H */
