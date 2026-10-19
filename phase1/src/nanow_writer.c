#include "nanow_format.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Packs 4 ternary elements into 1 byte (2-bit packing).
 * Mapping:
 *   0  -> 00b
 *  +1  -> 01b
 *  -1  -> 11b
 */
size_t pack_ternary_2bit(uint8_t *out_bytes, const int8_t *in_ternary, size_t count) {
    size_t out_len = (count + 3) / 4;
    memset(out_bytes, 0, out_len);

    for (size_t i = 0; i < count; ++i) {
        size_t byte_idx = i / 4;
        int shift = (i % 4) * 2;
        int8_t v = in_ternary[i];
        uint8_t code = 0;
        if (v == 1) {
            code = 0x01; // 01b
        } else if (v == -1) {
            code = 0x03; // 11b
        } else {
            code = 0x00; // 00b (0)
        }
        out_bytes[byte_idx] |= (code << shift);
    }
    return out_len;
}

/**
 * Packs 5 ternary elements into 1 byte (Radix-3 packing).
 * Values mapped to [0, 1, 2]: -1 -> 0, 0 -> 1, +1 -> 2
 * byte = v0 + 3*v1 + 9*v2 + 27*v3 + 81*v4 <= 242 (< 256)
 */
size_t pack_ternary_radix3(uint8_t *out_bytes, const int8_t *in_ternary, size_t count) {
    size_t out_len = (count + 4) / 5;
    memset(out_bytes, 0, out_len);

    for (size_t i = 0; i < count; i += 5) {
        size_t byte_idx = i / 5;
        uint32_t val = 0;
        uint32_t multiplier = 1;

        for (int k = 0; k < 5; ++k) {
            if (i + k < count) {
                int8_t v = in_ternary[i + k];
                uint32_t digit = (v == -1) ? 0 : ((v == 0) ? 1 : 2);
                val += digit * multiplier;
            } else {
                // padding with zero (digit 1)
                val += 1 * multiplier;
            }
            multiplier *= 3;
        }
        out_bytes[byte_idx] = (uint8_t)val;
    }
    return out_len;
}

/**
 * Calculates page-aligned padding.
 */
static size_t pad_to_alignment(size_t offset, size_t alignment) {
    size_t rem = offset % alignment;
    if (rem == 0) return 0;
    return alignment - rem;
}

/**
 * Writes a full .nanow file given model metadata and tensor buffers.
 */
int nanow_write_file(const char *filepath,
                     const nanow_header_t *hdr_in,
                     const nanow_tensor_desc_t *descs,
                     const void **tensor_payloads) {
    FILE *f = fopen(filepath, "wb");
    if (!f) return -1;

    nanow_header_t hdr = *hdr_in;
    hdr.magic = NANOW_MAGIC;
    hdr.version = NANOW_VERSION;

    size_t header_and_table_size = sizeof(nanow_header_t) + hdr.num_tensors * sizeof(nanow_tensor_desc_t);
    size_t initial_padding = pad_to_alignment(header_and_table_size, NANOW_PAGE_ALIGNMENT);
    size_t current_data_offset = header_and_table_size + initial_padding;

    // Allocate copy of descriptors to compute offsets
    nanow_tensor_desc_t *final_descs = (nanow_tensor_desc_t *)malloc(hdr.num_tensors * sizeof(nanow_tensor_desc_t));
    if (!final_descs) {
        fclose(f);
        return -2;
    }

    for (uint32_t i = 0; i < hdr.num_tensors; ++i) {
        final_descs[i] = descs[i];
        final_descs[i].data_offset = current_data_offset;
        size_t pad = pad_to_alignment(final_descs[i].data_size, NANOW_PAGE_ALIGNMENT);
        current_data_offset += final_descs[i].data_size + pad;
    }

    hdr.header_size = (uint32_t)header_and_table_size;
    hdr.file_size = current_data_offset;

    // Write Header
    fwrite(&hdr, sizeof(nanow_header_t), 1, f);
    // Write Descriptors
    fwrite(final_descs, sizeof(nanow_tensor_desc_t), hdr.num_tensors, f);

    // Write Initial Padding
    if (initial_padding > 0) {
        uint8_t zeros[NANOW_PAGE_ALIGNMENT] = {0};
        fwrite(zeros, 1, initial_padding, f);
    }

    // Write Tensor Payloads aligned
    for (uint32_t i = 0; i < hdr.num_tensors; ++i) {
        fwrite(tensor_payloads[i], 1, final_descs[i].data_size, f);
        size_t pad = pad_to_alignment(final_descs[i].data_size, NANOW_PAGE_ALIGNMENT);
        if (pad > 0) {
            uint8_t zeros[NANOW_PAGE_ALIGNMENT] = {0};
            fwrite(zeros, 1, pad, f);
        }
    }

    free(final_descs);
    fclose(f);
    return 0;
}
