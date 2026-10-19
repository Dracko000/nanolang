#ifndef NANOW_MMAP_H
#define NANOW_MMAP_H

#include "nanow_format.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int fd;                             /* File descriptor */
    void *mmap_base;                    /* Base virtual memory address mapped via mmap */
    size_t mmap_len;                    /* Total mapped byte size */
    nanow_header_t header;              /* Cached header */
    nanow_tensor_desc_t *tensors;       /* Pointer to tensor descriptors table */
    bool is_open;
} nanow_model_t;

/**
 * Opens a .nanow file and creates a zero-copy memory map.
 * Validates magic number, version, and page alignment.
 * Returns 0 on success, negative error code on failure.
 */
int nanow_model_open(nanow_model_t *model, const char *filepath);

/**
 * Closes the memory map and releases OS handles.
 */
void nanow_model_close(nanow_model_t *model);

/**
 * Searches for a tensor by its canonical name.
 * Returns pointer to descriptor, or NULL if not found.
 */
const nanow_tensor_desc_t *nanow_model_get_tensor_desc(const nanow_model_t *model, const char *name);

/**
 * Obtains the direct raw memory pointer to tensor payload in mmap space.
 * ZERO COPY: Returns a direct address in unified memory, no memory allocated.
 */
const void *nanow_model_get_tensor_data(const nanow_model_t *model, const nanow_tensor_desc_t *desc);

#ifdef __cplusplus
}
#endif

#endif /* NANOW_MMAP_H */
