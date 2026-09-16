#ifndef EDNX_GTCRN_ERB_H
#define EDNX_GTCRN_ERB_H

/* Fixed GTCRN ERB DSP, not a learned integer operator. One immutable CSR
 * table serves both directions. The caller owns the handle and all buffers;
 * no heap allocation, persistent audio state or transpose table is used.
 *
 * Payload:65 little-endian uint16 row offsets, nnz uint8 column indices,
 * then nnz IEEE754 little-endian float32 values, with no header or padding.
 * nnz is the final row offset. Rows are64 bands over192 high FFT bins.
 * The first65 bins are copied unchanged in either direction.
 */
#include <stddef.h>
#include <stdint.h>

typedef struct ednx_erb ednx_erb;

size_t ednx_erb_handle_bytes(void);
/* Exact payload length, sorted unique indices per row, nonzero finite
 * values and all offsets are validated. Failure clears the handle. The
 * payload may be unaligned, but must remain unchanged/alive after init. */
int ednx_erb_init(ednx_erb *model, const void *payload, size_t bytes);
size_t ednx_erb_nonzero_count(const ednx_erb *model);
/* Each channel is contiguous: forward channels*257 -> channels*129;
 * inverse channels*129 -> channels*257. GTCRN uses3 then2 channels.
 * Inputs/outputs must have exactly the declared sample counts and must be
 * disjoint from each other, the handle and payload. Nonfinite input/output
 * fails; output may be partially written on arithmetic overflow.
 * Scalar float32 summation can differ from dense BLAS summation order.
 */
int ednx_erb_forward(const ednx_erb *model, const float *input, size_t input_samples,
                     float *output, size_t output_samples, size_t channels);
int ednx_erb_inverse(const ednx_erb *model, const float *input, size_t input_samples,
                     float *output, size_t output_samples, size_t channels);

#endif
