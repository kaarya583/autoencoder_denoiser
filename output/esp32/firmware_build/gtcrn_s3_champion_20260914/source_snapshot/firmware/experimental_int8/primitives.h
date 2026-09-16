#ifndef EDNX_PRIMITIVES_H
#define EDNX_PRIMITIVES_H

/* Isolated numerical primitives. No GTCRN graph, firmware dispatch or model
 * format is implemented here. All buffers and handles remain caller-owned.
 * Parameter buffers must remain unchanged and alive after successful init.
 * Handles and INT32 arrays require their natural alignment. */
#include <stddef.h>
#include <stdint.h>

typedef struct { const void *data; size_t bytes; } ednx_buffer;
typedef struct ednx_gru ednx_gru;
typedef struct ednx_layer_norm ednx_layer_norm;

typedef struct {
    uint32_t input_size, hidden_size;
    int input_exponent, state_exponent, logit_exponent, accumulator_exponent;
    ednx_buffer weight_ih, weight_hh, bias_ih, bias_hh;
    ednx_buffer exponent_ih, exponent_hh, sigmoid_lut, tanh_lut;
} ednx_gru_spec;

typedef struct {
    /* Optional diagnostics. Both buffers must be disjoint from each other,
     * input/state/output/scratch, handles, and parameter arrays. Codes are six
     * contiguous hidden_size rows: reset/update/candidate logits, then
     * reset/update/candidate outputs. */
    int8_t *codes;
    size_t codes_bytes;
    int32_t *candidate_accumulators;
    size_t candidate_count;
} ednx_gru_trace;

typedef struct {
    uint32_t size;
    int input_exponent, gamma_exponent, output_exponent;
    uint32_t variance_fractional_bits;
    uint64_t epsilon_code; /* Already converted from real variance units. */
    ednx_buffer gamma, beta;
} ednx_layer_norm_spec;

/* Zero means success. Invalid contracts/arguments return -1. The init
 * routines validate every declared array length and worst-case arithmetic
 * bound before installing a handle. Failed init clears the handle. */
size_t ednx_gru_handle_bytes(void);
int ednx_gru_init(ednx_gru *model, const ednx_gru_spec *spec);
size_t ednx_gru_scratch_bytes(const ednx_gru *model);
/* Input/state/output lengths are input_size/hidden_size/hidden_size bytes.
 * Scratch needs hidden_size bytes, disjoint from input/state/output. Output
 * may alias state: all new values are staged before state is overwritten.
 * The model and scratch contain no persistent neural state. */
int ednx_gru_step(const ednx_gru *model, const int8_t *input,
                  const int8_t *state, int8_t *output,
                  int8_t *scratch, size_t scratch_bytes,
                  ednx_gru_trace *trace);

size_t ednx_layer_norm_handle_bytes(void);
int ednx_layer_norm_init(ednx_layer_norm *model, const ednx_layer_norm_spec *spec);
/* Exactly size INT8 values per call. Input/output may alias. Two passes,
 * scalar scratch, one wide reciprocal division per frame, no heap/state. */
int ednx_layer_norm_frame(const ednx_layer_norm *model,
                          const int8_t *input, int8_t *output);

#ifdef EDNX_TESTING
/* Arithmetic probes compiled only into the host test library. */
int ednx_test_round_divide(int64_t numerator, uint64_t denominator, int64_t *output);
uint64_t ednx_test_isqrt(uint64_t value);
#endif
#endif
