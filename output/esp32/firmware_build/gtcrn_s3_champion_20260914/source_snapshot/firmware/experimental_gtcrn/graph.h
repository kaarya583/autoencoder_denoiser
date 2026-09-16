#ifndef EDNG_GRAPH_H
#define EDNG_GRAPH_H

/* Portable fixed-topology GTCRN neural graph. The immutable packed model,
 * model handle, persistent state, workspace, input and output are all
 * caller-owned and mutually disjoint. The model blob and handle require
 * eight-byte alignment. No heap allocation occurs in this implementation.
 *
 * FFT, sparse ERB, complex multiplication and overlap-add belong to external
 * DSP. One call consumes [3,129] INT8 features and emits [2,129] signed Q7
 * complex-mask values. All persistent neural history is INT8. Initializing
 * the handle validates the packed topology and primitive arithmetic bounds.
 * This scalar C implementation is not evidence of ESP32 real-time speed. */
#include <stddef.h>
#include <stdint.h>

typedef struct edng_model edng_model;
size_t edng_handle_bytes(void);
size_t edng_state_bytes(void);
size_t edng_workspace_bytes(void);
int edng_init(edng_model *model, size_t handle_bytes, const void *packed, size_t packed_bytes);
int edng_reset(const edng_model *model, int8_t *state, size_t state_bytes);
int edng_input_exponent(const edng_model *model, int *exponent);
/* Any nonzero status is an error. Input/output sizes must be exactly387/258;
 * state and workspace may be larger than their advertised minimum. If a
 * primitive unexpectedly fails after a frame starts, discard/reset state. */
int edng_process_frame(const edng_model *model,
                        int8_t *state, size_t state_bytes,
                        const int8_t *input, size_t input_bytes,
                        int8_t *output, size_t output_bytes,
                        void *workspace, size_t workspace_bytes);

#endif
