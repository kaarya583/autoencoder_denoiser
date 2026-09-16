#ifndef EDNX_OPERATORS_H
#define EDNX_OPERATORS_H

/* Isolated scalar integer operators, with no model format or graph dispatch.
 * Parameters and handles remain caller-owned. All tensors are contiguous;
 * convolution layout is [channel,time,frequency], without a batch axis.
 * Buffers may not overlap unless an individual API explicitly permits it. */
#include "primitives.h"

enum { EDNX_LINEAR = 0, EDNX_CONV2D = 1, EDNX_CONV_TRANSPOSE2D = 2 };
typedef struct ednx_affine ednx_affine;
typedef struct {
    uint32_t kind, input_channels, output_channels, groups;
    uint32_t kernel_time, kernel_frequency, stride_time, stride_frequency;
    uint32_t padding_time, padding_frequency, dilation_time, dilation_frequency;
    uint32_t output_padding_time, output_padding_frequency;
    int input_exponent, output_exponent;
    /* Weights: [output,input/group,kT,kF], with no implicit kernel reversal.
     * Linear weights: [output,input]. Bias is naturally aligned INT32.
     * Exponents are one signed INT8 power-of-two exponent per output. */
    ednx_buffer weights, bias, exponents;
} ednx_affine_spec;

size_t ednx_affine_handle_bytes(void);
int ednx_affine_init(ednx_affine *model, const ednx_affine_spec *spec);
int ednx_affine_output_shape(const ednx_affine *model, uint32_t time, uint32_t frequency,
                             uint32_t *output_time, uint32_t *output_frequency);
/* Linear input/output is [time,features], frequency must be 1. Optional raw
 * accumulators follow output layout and require INT32 natural alignment. */
int ednx_affine_run(const ednx_affine *model, const int8_t *input, size_t input_bytes,
                     uint32_t time, uint32_t frequency, int8_t *output, size_t output_bytes,
                     int32_t *accumulators, size_t accumulator_count);

/* Causal wrapper around already-converted ordinary Conv2d weights: time
 * stride 1 and padding 0. History contains actual input-grid codes [C,h,F],
 * h=(kT-1)*dT. Frequency transform inserts stride-1 zeros after each sample,
 * then applies signed padding/cropping. Ordinary wrapper uses 1,0,0;
 * converted transpose uses its stored F_stride and asymmetric pad pair.
 * Workspace contains only transformed [C,h+1,F'] input. No heap allocation.
 * next_history may alias history; it must be disjoint from other buffers. */
size_t ednx_stream_history_bytes(const ednx_affine *model, uint32_t frequency);
size_t ednx_stream_workspace_bytes(const ednx_affine *model, uint32_t frequency,
                                   uint32_t frequency_stride, int padding_left, int padding_right);
int ednx_stream_conv(const ednx_affine *model, const int8_t *frame, size_t frame_bytes,
                      const int8_t *history, size_t history_bytes, uint32_t frequency,
                      uint32_t frequency_stride, int padding_left, int padding_right,
                      int8_t *output, size_t output_bytes,
                      int8_t *next_history, size_t next_history_bytes,
                      int8_t *workspace, size_t workspace_bytes);

/* Power-of-two activation exponents are -16..8. Vector operations allow
 * exact in-place input/output; residual allows either input to alias output.
 * Caller supplies at least count elements for every vector. */
int ednx_regrid(const int8_t *input, int8_t *output, size_t count, int input_exponent, int output_exponent);
int ednx_residual(const int8_t *left, const int8_t *right, int8_t *output, size_t count,
                    int left_exponent, int right_exponent, int output_exponent);
/* Layout [channels,inner]. A slope can be shared or supplied per channel. */
int ednx_prelu(const int8_t *input, int8_t *output, uint32_t channels, size_t inner,
                const int8_t *slopes, size_t slope_count, int slope_exponent,
                int input_exponent, int output_exponent);
int ednx_lut(const int8_t *input, int8_t *output, size_t count, const int8_t table[256]);
/* Energy input [rows,frequency], output [rows], frequency 1..1024.
 * Product input/output [rows,frequency]; one probability code per row.
 * Probabilities use (signed_code+128)/255, with both endpoints represented. */
int ednx_attention_energy(const int8_t *input, int8_t *output, size_t rows, uint32_t frequency,
                            int input_exponent, int output_exponent);
int ednx_attention_product(const int8_t *input, const int8_t *probabilities, int8_t *output,
                             size_t rows, uint32_t frequency, int input_exponent, int output_exponent);
int ednx_subband(const int8_t *input, int8_t *output, uint32_t channels, uint32_t time, uint32_t frequency);
int ednx_shuffle(const int8_t *left, const int8_t *right, int8_t *output, uint32_t channels, size_t inner);

#endif
