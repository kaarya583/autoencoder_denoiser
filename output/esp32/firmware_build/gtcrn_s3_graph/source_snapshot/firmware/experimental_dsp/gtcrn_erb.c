#include "gtcrn_erb.h"
#include <float.h>
#include <math.h>
#include <string.h>

#if FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128
#error "GTCRN ERB requires IEEE754 float32"
#endif
typedef char float_must_be_four_bytes[sizeof(float) == 4 ? 1 : -1];

#define ERB_MAGIC UINT32_C(0x58455242)
#define BANDS 64u
#define HIGH_BINS 192u
#define LOW_BINS 65u
#define FFT_BINS 257u
#define ERB_BINS 129u
#define OFFSETS_BYTES 130u

struct ednx_erb {
    uint32_t magic;
    uint16_t nonzero;
    const uint8_t *offsets, *indices, *values;
};

static uint16_t read_u16(const uint8_t *value) {
    return (uint16_t)((uint16_t)value[0] | ((uint16_t)value[1] << 8));
}

static float read_f32(const uint8_t *value) {
    uint32_t bits = (uint32_t)value[0] | ((uint32_t)value[1] << 8)
                  | ((uint32_t)value[2] << 16) | ((uint32_t)value[3] << 24);
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

size_t ednx_erb_handle_bytes(void) { return sizeof(ednx_erb); }

int ednx_erb_init(ednx_erb *model, const void *payload, size_t bytes) {
    if (!model) return -1;
    memset(model, 0, sizeof(*model));
    if (!payload || bytes < OFFSETS_BYTES) return -1;
    const uint8_t *data = payload;
    uint16_t nonzero = read_u16(data + 2u * BANDS);
    if (nonzero > BANDS * HIGH_BINS || bytes != OFFSETS_BYTES + 5u * nonzero || read_u16(data) != 0) return -1;
    const uint8_t *indices = data + OFFSETS_BYTES;
    const uint8_t *values = indices + nonzero;
    for (size_t row = 0; row < BANDS; ++row) {
        uint16_t start = read_u16(data + 2u * row), end = read_u16(data + 2u * (row + 1u));
        if (end < start || end > nonzero) return -1;
        for (size_t index = start; index < end; ++index) {
            float coefficient = read_f32(values + 4u * index);
            if (indices[index] >= HIGH_BINS || (index > start && indices[index] <= indices[index - 1u]) ||
                !isfinite(coefficient) || coefficient == 0.0f) return -1;
        }
    }
    model->offsets = data; model->indices = indices; model->values = values;
    model->nonzero = nonzero; model->magic = ERB_MAGIC;
    return 0;
}

size_t ednx_erb_nonzero_count(const ednx_erb *model) {
    return model && model->magic == ERB_MAGIC ? model->nonzero : 0;
}

static int validate(const ednx_erb *model, const float *input, size_t input_samples,
                     float *output, size_t output_samples, size_t channels, size_t in, size_t out) {
    if (!model || model->magic != ERB_MAGIC || !input || !output || !channels ||
        channels > SIZE_MAX / (FFT_BINS * sizeof(float)) ||
        input_samples != channels * in || output_samples != channels * out) return -1;
    uintptr_t first = (uintptr_t)input, second = (uintptr_t)output;
    size_t input_bytes = input_samples * sizeof(float), output_bytes = output_samples * sizeof(float);
    if (first > UINTPTR_MAX - input_bytes || second > UINTPTR_MAX - output_bytes ||
        (first < second + output_bytes && second < first + input_bytes)) return -1;
    for (size_t index = 0; index < input_samples; ++index) if (!isfinite(input[index])) return -1;
    return 0;
}

int ednx_erb_forward(const ednx_erb *model, const float *input, size_t input_samples,
                     float *output, size_t output_samples, size_t channels) {
    if (validate(model, input, input_samples, output, output_samples, channels, FFT_BINS, ERB_BINS)) return -1;
    for (size_t channel = 0; channel < channels; ++channel) {
        const float *source = input + channel * FFT_BINS;
        float *result = output + channel * ERB_BINS;
        memcpy(result, source, LOW_BINS * sizeof(float));
        for (size_t row = 0; row < BANDS; ++row) {
            float total = 0.0f;
            size_t start = read_u16(model->offsets + 2u * row), end = read_u16(model->offsets + 2u * (row + 1u));
            for (size_t index = start; index < end; ++index)
                total += read_f32(model->values + 4u * index) * source[LOW_BINS + model->indices[index]];
            if (!isfinite(total)) return -1;
            result[LOW_BINS + row] = total;
        }
    }
    return 0;
}

int ednx_erb_inverse(const ednx_erb *model, const float *input, size_t input_samples,
                     float *output, size_t output_samples, size_t channels) {
    if (validate(model, input, input_samples, output, output_samples, channels, ERB_BINS, FFT_BINS)) return -1;
    for (size_t channel = 0; channel < channels; ++channel) {
        const float *source = input + channel * ERB_BINS;
        float *result = output + channel * FFT_BINS;
        memcpy(result, source, LOW_BINS * sizeof(float));
        memset(result + LOW_BINS, 0, HIGH_BINS * sizeof(float));
        for (size_t row = 0; row < BANDS; ++row) {
            size_t start = read_u16(model->offsets + 2u * row), end = read_u16(model->offsets + 2u * (row + 1u));
            for (size_t index = start; index < end; ++index) {
                size_t bin = LOW_BINS + model->indices[index];
                result[bin] += read_f32(model->values + 4u * index) * source[LOW_BINS + row];
            }
        }
        for (size_t bin = LOW_BINS; bin < FFT_BINS; ++bin) if (!isfinite(result[bin])) return -1;
    }
    return 0;
}
