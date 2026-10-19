#ifndef NANOW_FORMAT_H
#define NANOW_FORMAT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NANOW_MAGIC            0x4E414E4F  /* 'N', 'A', 'N', 'O' */
#define NANOW_VERSION          1
#define NANOW_PAGE_ALIGNMENT   4096        /* 4KB Page boundary for zero-copy mmap */
#define NANOW_MAX_NAME_LEN     64

/* Header feature flags (nanow_header_t.flags).
 *
 * Model containers leave this at 0. A container whose payload is a KV cache
 * rather than model weights sets NANOW_FLAG_KV_CACHE, so a reader can refuse to
 * treat cache state as a model instead of guessing from the file name.
 * phase1/python/export_to_nanow.py mirrors this value; keep the two in step. */
#define NANOW_FLAG_KV_CACHE    0x0001

/* Packing formats */
typedef enum {
    NANOW_PACKING_RAW_INT8 = 0, /* Unpacked int8: {-1, 0, 1} (8 bits / weight) */
    NANOW_PACKING_2BIT     = 1, /* 2-bit aligned: 4 ternary weights per byte */
    NANOW_PACKING_RADIX3   = 2  /* Radix-3: 5 ternary weights per byte (1.6 bits / weight) */
} nanow_packing_t;

/* Tensor dtype */
typedef enum {
    NANOW_DTYPE_TERNARY = 0,    /* Discrete {-1, 0, 1} */
    NANOW_DTYPE_INT8    = 1,    /* Activations / Embedding */
    NANOW_DTYPE_FP16    = 2,    /* Scalar scales or critical RMSNorm weights */
    NANOW_DTYPE_FP32    = 3
} nanow_dtype_t;

#pragma pack(push, 1)

/* File Header (Aligned to 64 bytes) */
typedef struct {
    uint32_t magic;             /* NANOW_MAGIC */
    uint32_t version;           /* File format version (1) */
    uint32_t flags;             /* Feature flags */
    uint32_t num_tensors;       /* Total number of tensor records */
    
    /* Model Architecture Metadata */
    uint32_t num_layers;        /* Number of layers */
    uint32_t hidden_dim;        /* Model hidden dimension (e.g. 2048) */
    uint32_t state_dim;         /* SSM recurrent state dimension N (e.g. 16) */
    uint32_t vocab_size;        /* Vocabulary size (e.g. 32000) */
    
    uint64_t total_params;      /* Total model parameters count */
    uint64_t file_size;         /* Total container file size in bytes */
    uint32_t header_size;       /* Size of header + tensor directory */
    uint32_t reserved[3];       /* Padding / future extensions */
} nanow_header_t;

/* Tensor Metadata Descriptor */
typedef struct {
    char name[NANOW_MAX_NAME_LEN]; /* e.g., "layers.0.ssm.in_proj" */
    uint32_t dtype;                /* nanow_dtype_t */
    uint32_t packing;              /* nanow_packing_t */
    uint32_t ndims;                /* Number of dimensions (1 to 4) */
    uint32_t shape[4];             /* Dimensions: [dim0, dim1, dim2, dim3] */
    uint64_t num_elements;         /* Total number of values */
    uint64_t data_offset;          /* 4096-byte aligned byte offset from start of file */
    uint64_t data_size;            /* Compressed/packed payload size in bytes */
    float scale_factor;            /* Scaling parameter gamma or norm alpha */
} nanow_tensor_desc_t;

#pragma pack(pop)

#ifdef __cplusplus
}
#endif

#endif /* NANOW_FORMAT_H */
