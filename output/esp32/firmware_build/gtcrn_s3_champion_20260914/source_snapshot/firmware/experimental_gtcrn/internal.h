#ifndef EDNG_INTERNAL_H
#define EDNG_INTERNAL_H

#include "graph.h"
#include "layout.h"
#include "../experimental_int8/operators.h"

#define EDNG_MAGIC UINT32_C(0x47544938)
#define EDNG_NEURAL_STATE_BYTES 18048u

/* The strict packed loader initializes native handles into the caller's
 * trailing handle arena, aligned to8 bytes. array is used only for the
 * scalar PReLU slope or256-entry activation table; handles are used only
 * for affine/stream, GRU and LayerNorm. No parameter array is copied. */
typedef struct {
    void *handle;
    const int8_t *array;
    int input_exponent, output_exponent, auxiliary;
} edng_operator;

struct edng_model {
    uint32_t magic;
    const uint8_t *packed;
    size_t packed_bytes;
    const int8_t *grids;
    edng_operator operators[EDNG_RECORDS];
};

#endif
