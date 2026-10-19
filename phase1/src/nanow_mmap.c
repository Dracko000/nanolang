#include "nanow_mmap.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string.h>

int nanow_model_open(nanow_model_t *model, const char *filepath) {
    if (!model || !filepath) return -1;
    memset(model, 0, sizeof(nanow_model_t));

    model->fd = open(filepath, O_RDONLY);
    if (model->fd < 0) {
        return -2;
    }

    struct stat sb;
    if (fstat(model->fd, &sb) < 0) {
        close(model->fd);
        return -3;
    }
    model->mmap_len = (size_t)sb.st_size;

    if (model->mmap_len < sizeof(nanow_header_t)) {
        close(model->fd);
        return -4;
    }

    // Direct zero-copy memory map (Read-only shared)
    model->mmap_base = mmap(NULL, model->mmap_len, PROT_READ, MAP_SHARED, model->fd, 0);
    if (model->mmap_base == MAP_FAILED) {
        close(model->fd);
        return -5;
    }

    // Read and validate Header
    const nanow_header_t *hdr = (const nanow_header_t *)model->mmap_base;
    if (hdr->magic != NANOW_MAGIC) {
        munmap(model->mmap_base, model->mmap_len);
        close(model->fd);
        return -6;
    }

    if (hdr->version != NANOW_VERSION) {
        munmap(model->mmap_base, model->mmap_len);
        close(model->fd);
        return -7;
    }

    model->header = *hdr;
    model->tensors = (nanow_tensor_desc_t *)((uint8_t *)model->mmap_base + sizeof(nanow_header_t));
    model->is_open = true;

    return 0;
}

void nanow_model_close(nanow_model_t *model) {
    if (!model || !model->is_open) return;
    if (model->mmap_base && model->mmap_base != MAP_FAILED) {
        munmap(model->mmap_base, model->mmap_len);
    }
    if (model->fd >= 0) {
        close(model->fd);
    }
    memset(model, 0, sizeof(nanow_model_t));
}

const nanow_tensor_desc_t *nanow_model_get_tensor_desc(const nanow_model_t *model, const char *name) {
    if (!model || !model->is_open || !name) return NULL;

    for (uint32_t i = 0; i < model->header.num_tensors; ++i) {
        if (strncmp(model->tensors[i].name, name, NANOW_MAX_NAME_LEN) == 0) {
            return &model->tensors[i];
        }
    }
    return NULL;
}

const void *nanow_model_get_tensor_data(const nanow_model_t *model, const nanow_tensor_desc_t *desc) {
    if (!model || !model->is_open || !desc) return NULL;

    // Check bounds
    if (desc->data_offset + desc->data_size > model->mmap_len) {
        return NULL;
    }

    // Zero-copy: Direct pointer computation into mapped address space
    return (const void *)((const uint8_t *)model->mmap_base + desc->data_offset);
}
