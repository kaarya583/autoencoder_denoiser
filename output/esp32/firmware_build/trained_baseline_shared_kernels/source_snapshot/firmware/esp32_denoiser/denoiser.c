#include "denoiser.h"
#include "integer_kernels.h"

#include <limits.h>
#include <string.h>

#ifdef ESP_PLATFORM
#include "sdkconfig.h"
#ifdef CONFIG_EDN_S3_SIMD_DOT
#define EDN_USE_S3_SIMD 1
#endif
#endif
#ifndef EDN_USE_S3_SIMD
#define EDN_USE_S3_SIMD 0
#endif

#if EDN_USE_S3_SIMD
#define EDN_SIMD_INPUT_MAX 512u
#define EDN_SIMD_GUARD 32u
#define EDN_SIMD_TEST_MAX 1056u
/* Pinned ESP-NN raw signed dot ABI. The two length units intentionally differ. */
extern int32_t esp_nn_dot_s8_aligned_esp32s3(const int8_t *, const int8_t *, int bytes);
extern int32_t esp_nn_dot_s8_unaligned_esp32s3(const int8_t *, const int8_t *, int blocks);
#endif

#define EDN_HEADER_BYTES 32u
#define EDN_LAYER_BYTES 24u

static uint16_t read_u16(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t read_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int32_t read_i32(const uint8_t *p) {
    uint32_t value = read_u32(p);
    return value <= INT32_MAX ? (int32_t)value : -1 - (int32_t)(~value);
}

static void write_u32(uint8_t *p, uint32_t value) {
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
    p[2] = (uint8_t)(value >> 16);
    p[3] = (uint8_t)(value >> 24);
}

static int8_t clamp_i8(int64_t value) {
    return value < -128 ? -128 : value > 127 ? 127 : (int8_t)value;
}

static int8_t requantize(int32_t value, int shift) {
    int64_t magnitude;
    if (shift >= 0) {
        if (shift >= 7) return value < 0 ? -128 : value > 0 ? 127 : 0;
        return clamp_i8((int64_t)value * ((int64_t)1 << shift));
    }
    if (shift < -32) return 0;
    magnitude = value < 0 ? -(int64_t)value : value;
    magnitude = (magnitude + ((int64_t)1 << (-shift - 1))) >> -shift;
    return clamp_i8(value < 0 ? -magnitude : magnitude);
}

static const uint8_t *layer_at(const edn_model *m, unsigned index) {
    return m->data + EDN_HEADER_BYTES + EDN_LAYER_BYTES * index;
}

static int region_valid(size_t offset, size_t bytes, size_t length) {
    return offset <= length && bytes <= length - offset;
}

int edn_init(edn_model *m, const void *source, size_t length) {
    const uint8_t *data = (const uint8_t *)source;
    size_t table_end, last_offset = 0, dsp_offset;
    uint64_t state_bytes;
    unsigned i, layers;
    edn_model parsed;
    if (!m || !data || length < EDN_HEADER_BYTES ||
        memcmp(data, "EDNSI8\0\0", 8) || read_u32(data + 8) != 2 ||
        read_u32(data + 24) != length || data[23] != 1) return -1;
    memset(&parsed, 0, sizeof(parsed));
    parsed.data = data;
    parsed.model_bytes = length;
    parsed.input_channels = read_u16(data + 12);
    parsed.hidden_channels = read_u16(data + 14);
    parsed.output_channels = read_u16(data + 16);
    parsed.blocks = read_u16(data + 18);
    parsed.input_exponent = (int8_t)data[20];
    parsed.hidden_exponent = (int8_t)data[21];
    parsed.output_exponent = (int8_t)data[22];
    parsed.state_bytes = read_u32(data + 28);
    if (!parsed.input_channels || !parsed.hidden_channels || !parsed.output_channels ||
        parsed.blocks > 32 || parsed.input_exponent < -16 || parsed.input_exponent > 8 ||
        parsed.hidden_exponent < -12 || parsed.hidden_exponent > 0 || parsed.output_exponent != -7) return -1;
    layers = 2u + 2u * parsed.blocks;
    table_end = EDN_HEADER_BYTES + EDN_LAYER_BYTES * layers;
    if (table_end > length) return -1;
    state_bytes = 4u * parsed.blocks + 3u * parsed.hidden_channels;
    for (i = 0; i < layers; ++i) {
        const uint8_t *l = layer_at(&parsed, i);
        unsigned kind = l[0], kernel = l[3], in = read_u16(l + 4), out = read_u16(l + 6);
        unsigned dilation = read_u16(l + 8), expected_kind = i > 0 && i < layers - 1 && (i & 1u);
        size_t terms = kind ? kernel : in;
        size_t weight_offset = read_u32(l + 12), bias_offset = read_u32(l + 16), exponent_offset = read_u32(l + 20);
        unsigned row;
        if (kind != expected_kind || kernel != (kind ? 3u : 1u) || !dilation ||
            in != (i == 0 ? parsed.input_channels : parsed.hidden_channels) ||
            out != (i == layers - 1 ? parsed.output_channels : parsed.hidden_channels) ||
            (int8_t)l[1] != (i == 0 ? parsed.input_exponent : parsed.hidden_exponent) ||
            (int8_t)l[2] != (i == layers - 1 ? parsed.output_exponent : parsed.hidden_exponent) ||
            weight_offset < table_end || bias_offset < table_end || exponent_offset < table_end ||
            !region_valid(weight_offset, terms * out, length) ||
            !region_valid(bias_offset, 4u * out, length) ||
            !region_valid(exponent_offset, out, length)) return -1;
        if (weight_offset + terms * out > last_offset) last_offset = weight_offset + terms * out;
        if (bias_offset + 4u * out > last_offset) last_offset = bias_offset + 4u * out;
        if (exponent_offset + out > last_offset) last_offset = exponent_offset + out;
        for (row = 0; row < out; ++row) {
            int exponent = (int8_t)data[exponent_offset + row];
            int32_t bias = read_i32(data + bias_offset + 4u * row);
            int64_t bound = bias < 0 ? -(int64_t)bias : bias;
            size_t k;
            if (exponent < -24 || exponent > 16) return -1;
            for (k = 0; k < terms; ++k) {
                int weight = (int8_t)data[weight_offset + row * terms + k];
                bound += 128 * (weight < 0 ? -weight : weight);
            }
            if (bound > INT32_MAX) return -1;
        }
        if (kind) state_bytes += (uint64_t)2 * dilation * parsed.hidden_channels;
    }
    if (state_bytes > UINT32_MAX || state_bytes != parsed.state_bytes) return -1;
    dsp_offset = (last_offset + 3u) & ~(size_t)3u;
    if (!region_valid(dsp_offset, 3988u, length) || dsp_offset + 3988u != length ||
        memcmp(data + dsp_offset, "DSP1", 4) || read_u16(data + dsp_offset + 4) != 16000 ||
        read_u16(data + dsp_offset + 6) != 512 || read_u16(data + dsp_offset + 8) != 256 ||
        read_u16(data + dsp_offset + 10) != 65 || read_u16(data + dsp_offset + 12) != 64) return -1;
    parsed.dsp_data = data + dsp_offset;
    parsed.dsp_bytes = length - dsp_offset;
    *m = parsed;
    return 0;
}

int edn_reset(const edn_model *m, void *state, size_t state_length) {
    if (!m || !m->data || !state || state_length < m->state_bytes) return -1;
    memset(state, 0, m->state_bytes);
    return 0;
}

static int32_t scalar_dot(const int8_t *a, const int8_t *b, unsigned length) {
    int32_t sum = 0;
    unsigned i;
    for (i = 0; i < length; ++i) sum += (int32_t)a[i] * b[i];
    return sum;
}

int32_t edn_dot_product(const int8_t *input, const int8_t *weights, unsigned length,
                        const uint8_t *weight_storage, size_t storage_bytes) {
#if EDN_USE_S3_SIMD
    unsigned full=length & ~15u;
    uintptr_t start=(uintptr_t)weight_storage, address=(uintptr_t)weights;
    uintptr_t alignment=address & 15u;
    if(full && !((uintptr_t)input & 15u) && address>=start &&
       region_valid((size_t)(address-start),length,storage_bytes)) {
        int32_t dot;
        if(!alignment) {
            dot=esp_nn_dot_s8_aligned_esp32s3(input,weights,(int)full);
        } else if(address-start>=alignment &&
                  region_valid((size_t)(address-start),full+EDN_SIMD_GUARD,storage_bytes)) {
            dot=esp_nn_dot_s8_unaligned_esp32s3(input,weights,(int)(full/16));
        } else return scalar_dot(input,weights,length);
        return dot+scalar_dot(input+full,weights+full,length-full);
    }
#else
    (void)weight_storage; (void)storage_bytes;
#endif
    return scalar_dot(input,weights,length);
}

const char *edn_neural_backend(void) {
    return EDN_USE_S3_SIMD ? "s3_simd_dot_with_scalar_fallback" : "portable_scalar";
}

int edn_backend_self_test(void) {
#if EDN_USE_S3_SIMD
    int8_t a[EDN_SIMD_TEST_MAX + EDN_SIMD_GUARD] __attribute__((aligned(16)));
    int8_t b[EDN_SIMD_TEST_MAX + EDN_SIMD_GUARD + 16] __attribute__((aligned(16)));
    static const unsigned lengths[] = {16, 17, 31, 32, 47, 64, 127, 384, 387, 511, 512, 1055, 1056};
    unsigned pattern, i, j, offset;
    for (pattern = 0; pattern < 4; ++pattern) {
        for (i = 0; i < sizeof(a); ++i)
            a[i] = pattern == 0 ? -128 : pattern == 1 ? 127 : (int8_t)((i * 71u + 19u) % 256u - 128);
        for (i = 0; i < sizeof(b); ++i)
            b[i] = pattern == 0 ? 127 : pattern == 1 ? 127 : (int8_t)((i * 43u + pattern * 31u) % 256u - 128);
        for (j = 0; j < sizeof(lengths) / sizeof(lengths[0]); ++j) {
            unsigned length = lengths[j], full = length & ~15u;
            int32_t aligned = esp_nn_dot_s8_aligned_esp32s3(a, b, (int)full);
            aligned += scalar_dot(a + full, b + full, length - full);
            if (aligned != scalar_dot(a, b, length)) return -1;
            for (offset = 0; offset < 16; ++offset) {
                int32_t unaligned = esp_nn_dot_s8_unaligned_esp32s3(a, b + offset, (int)(full / 16));
                unaligned += scalar_dot(a + full, b + offset + full, length - full);
                if (unaligned != scalar_dot(a, b + offset, length)) return -1;
            }
        }
    }
#endif
    return 0;
}

static void dense(const edn_model *m, const uint8_t *l, const int8_t *input, int8_t *output) {
    unsigned in = read_u16(l + 4), out = read_u16(l + 6), row;
    const int8_t *weights = (const int8_t *)(m->data + read_u32(l + 12));
    const uint8_t *bias = m->data + read_u32(l + 16);
    const int8_t *exponents = (const int8_t *)(m->data + read_u32(l + 20));
#if EDN_USE_S3_SIMD
    int8_t aligned_input[EDN_SIMD_INPUT_MAX + EDN_SIMD_GUARD] __attribute__((aligned(16)));
    unsigned full = in >= 16 && in <= EDN_SIMD_INPUT_MAX ? in & ~15u : 0;
    if (full) {
        memcpy(aligned_input, input, in);
        /* The unaligned kernel pipelines one additional input vector. */
        memset(aligned_input + in, 0, EDN_SIMD_GUARD);
    }
#endif
    for (row = 0; row < out; ++row) {
        const int8_t *row_weights = weights + (size_t)row * in;
        int32_t dot;
#if EDN_USE_S3_SIMD
        if(full) dot=edn_dot_product(aligned_input,row_weights,in,m->data,m->model_bytes);
        else
#endif
        dot=scalar_dot(input,row_weights,in);
        output[row] = requantize(read_i32(bias + 4u * row) + dot,
                                  (int8_t)l[1] + exponents[row] - (int8_t)l[2]);
    }
}

int edn_process_frame(const edn_model *m, void *state, size_t state_length,
                      const int8_t *features, int8_t *output) {
    uint8_t *positions = (uint8_t *)state;
    int8_t *history, *x, *y, *residual;
    unsigned block, channel, width;
    int minimum, maximum;
    size_t history_offset;
    if (!m || !m->data || !state || state_length < m->state_bytes || !features || !output) return -1;
    width = m->hidden_channels;
    x = (int8_t *)state + m->state_bytes - 3u * width;
    y = x + width;
    residual = y + width;
    maximum = 6 * (1 << -m->hidden_exponent);
    minimum = -maximum;
    if (minimum < -128) minimum = -128;
    if (maximum > 127) maximum = 127;
    dense(m, layer_at(m, 0), features, x);
    for (channel = 0; channel < width; ++channel) x[channel] = x[channel] < minimum ? (int8_t)minimum : x[channel] > maximum ? (int8_t)maximum : x[channel];
    history_offset = 4u * m->blocks;
    for (block = 0; block < m->blocks; ++block) {
        const uint8_t *dw = layer_at(m, 1u + 2u * block);
        unsigned dilation = read_u16(dw + 8), frames = 2u * dilation;
        unsigned position = read_u32(positions + 4u * block);
        unsigned middle;
        const int8_t *weights = (const int8_t *)(m->data + read_u32(dw + 12));
        const uint8_t *bias = m->data + read_u32(dw + 16);
        const int8_t *exponents = (const int8_t *)(m->data + read_u32(dw + 20));
        if (position >= frames) return -1;
        middle = (position + dilation) % frames;
        history = (int8_t *)state + history_offset;
        for (channel = 0; channel < width; ++channel) {
            int32_t acc = read_i32(bias + 4u * channel);
            acc += (int32_t)weights[3u * channel] * history[position * width + channel];
            acc += (int32_t)weights[3u * channel + 1u] * history[middle * width + channel];
            acc += (int32_t)weights[3u * channel + 2u] * x[channel];
            y[channel] = requantize(acc, (int8_t)dw[1] + exponents[channel] - (int8_t)dw[2]);
            if (y[channel] < 0) y[channel] = 0;
            if (y[channel] > maximum) y[channel] = (int8_t)maximum;
        }
        memcpy(history + position * width, x, width);
        write_u32(positions + 4u * block, (position + 1u) % frames);
        dense(m, layer_at(m, 2u + 2u * block), y, residual);
        for (channel = 0; channel < width; ++channel) {
            int32_t sum = (int32_t)x[channel] + residual[channel];
            x[channel] = sum < minimum ? (int8_t)minimum : sum > maximum ? (int8_t)maximum : (int8_t)sum;
        }
        history_offset += (size_t)frames * width;
    }
    dense(m, layer_at(m, 1u + 2u * m->blocks), x, output);
    return 0;
}
