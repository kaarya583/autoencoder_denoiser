#include "primitives.h"
#include <limits.h>
#include <string.h>

#define GRU_MAGIC UINT32_C(0x58475255)
#define LN_MAGIC UINT32_C(0x584c4e31)
#define MAX_SQUARED_DENOMINATOR UINT64_C(4611686014132420609)

struct ednx_gru {
    uint32_t magic, input_size, hidden_size;
    int input_exponent, logit_exponent;
    const int8_t *wi, *wh, *ei, *eh, *sigmoid, *tanh;
    const int32_t *bi, *bh;
};

struct ednx_layer_norm {
    uint32_t magic, size, bits;
    int shift;
    uint64_t epsilon;
    const int8_t *gamma;
    const int32_t *beta;
};

static uint64_t magnitude(int64_t value) {
    return value < 0 ? (uint64_t)(-(value + 1)) + 1u : (uint64_t)value;
}

static int8_t clip8(int64_t value) {
    return value < -128 ? -128 : (value > 127 ? 127 : (int8_t)value);
}

/* Callers establish |value| and shift bounds before entering this helper. */
static int64_t shift_round(int64_t value, int shift) {
    if (shift >= 0) return value * (INT64_C(1) << shift);
    uint64_t result = (magnitude(value) + (UINT64_C(1) << (-shift - 1))) >> -shift;
    return value < 0 ? -(int64_t)result : (int64_t)result;
}

static int64_t divide255(int64_t value) {
    uint64_t result = (magnitude(value) + 127u) / 255u;
    return value < 0 ? -(int64_t)result : (int64_t)result;
}

/* Portable high half of a 64x64 product, using only 32-bit limbs. Each
 * displayed intermediate fits uint64_t; no compiler __int128 extension. */
static uint64_t multiply_high(uint64_t a, uint64_t b) {
    uint64_t a0 = (uint32_t)a, a1 = a >> 32;
    uint64_t b0 = (uint32_t)b, b1 = b >> 32;
    uint64_t low = a0 * b0;
    uint64_t middle = a1 * b0 + (low >> 32);
    uint64_t carry = middle >> 32;
    middle = (uint32_t)middle + a0 * b1;
    return a1 * b1 + carry + (middle >> 32);
}

/* reciprocal=floor((2^64-1)/d). The high product underestimates floor(n/d)
 * by at most one for any uint64 n. One exact remainder correction suffices. */
static uint64_t reciprocal_divide(uint64_t n, uint64_t d, uint64_t reciprocal) {
    uint64_t q = multiply_high(n, reciprocal);
    if (n - q * d >= d) ++q;
    return q;
}

static int64_t round_divide(int64_t n, uint64_t d, uint64_t reciprocal) {
    uint64_t q = reciprocal_divide(magnitude(n) + d / 2u, d, reciprocal);
    return n < 0 ? -(int64_t)q : (int64_t)q;
}

/* Restoring integer square root; independent of Python/Torch algorithms. */
static uint64_t integer_sqrt(uint64_t value) {
    uint64_t result = 0, bit = UINT64_C(1) << 62;
    while (bit > value) bit >>= 2;
    while (bit) {
        if (value >= result + bit) {
            value -= result + bit;
            result = (result >> 1) + bit;
        } else result >>= 1;
        bit >>= 2;
    }
    return result;
}

static int buffer_is(ednx_buffer buffer, size_t bytes) {
    return buffer.data != NULL && buffer.bytes == bytes;
}

static int aligned_i32(ednx_buffer buffer) {
    return ((uintptr_t)buffer.data % sizeof(int32_t)) == 0;
}

size_t ednx_gru_handle_bytes(void) { return sizeof(ednx_gru); }

int ednx_gru_init(ednx_gru *model, const ednx_gru_spec *s) {
    if (!model) return -1;
    memset(model, 0, sizeof(*model));
    if (!s || s->input_size < 1 || s->input_size > 512 ||
        s->hidden_size < 1 || s->hidden_size > 128 ||
        s->input_exponent < -12 || s->input_exponent > 0 ||
        s->state_exponent != -7 || s->accumulator_exponent != -12 ||
        s->logit_exponent < -6 || s->logit_exponent > -2) return -1;
    size_t rows = 3u * s->hidden_size;
    if (!buffer_is(s->weight_ih, rows * s->input_size) ||
        !buffer_is(s->weight_hh, rows * s->hidden_size) ||
        !buffer_is(s->bias_ih, rows * sizeof(int32_t)) ||
        !buffer_is(s->bias_hh, rows * sizeof(int32_t)) ||
        !buffer_is(s->exponent_ih, rows) || !buffer_is(s->exponent_hh, rows) ||
        !buffer_is(s->sigmoid_lut, 256) || !buffer_is(s->tanh_lut, 256) ||
        !aligned_i32(s->bias_ih) || !aligned_i32(s->bias_hh)) return -1;
    const int8_t *wi = s->weight_ih.data, *wh = s->weight_hh.data;
    const int8_t *ei = s->exponent_ih.data, *eh = s->exponent_hh.data;
    const int32_t *bi = s->bias_ih.data, *bh = s->bias_hh.data;
    for (size_t row = 0; row < rows; ++row) {
        if (ei[row] < -20 || ei[row] > 4 || eh[row] < -20 || eh[row] > 4) return -1;
        uint64_t ib = magnitude(bi[row]), hb = magnitude(bh[row]);
        for (size_t j = 0; j < s->input_size; ++j) ib += 128u * magnitude(wi[row * s->input_size + j]);
        for (size_t j = 0; j < s->hidden_size; ++j) hb += 128u * magnitude(wh[row * s->hidden_size + j]);
        if (ib > INT32_MAX || hb > INT32_MAX) return -1;
        int64_t aligned = shift_round((int64_t)ib, s->input_exponent + ei[row] + 12)
                        + shift_round((int64_t)hb, eh[row] + 5);
        if (aligned > INT32_MAX) return -1;
    }
    model->input_size = s->input_size; model->hidden_size = s->hidden_size;
    model->input_exponent = s->input_exponent; model->logit_exponent = s->logit_exponent;
    model->wi = wi; model->wh = wh; model->bi = bi; model->bh = bh;
    model->ei = ei; model->eh = eh;
    model->sigmoid = s->sigmoid_lut.data; model->tanh = s->tanh_lut.data;
    model->magic = GRU_MAGIC;
    return 0;
}

size_t ednx_gru_scratch_bytes(const ednx_gru *model) {
    return model && model->magic == GRU_MAGIC ? model->hidden_size : 0;
}

static int32_t affine(const int8_t *x, const int8_t *weight, int32_t bias,
                       uint32_t count, int shift) {
    int32_t total = bias;
    for (uint32_t j = 0; j < count; ++j) total += (int32_t)x[j] * weight[j];
    return (int32_t)shift_round(total, shift);
}

static int8_t gru_logit(const ednx_gru *model, int32_t value) {
    return clip8(shift_round(value, -12 - model->logit_exponent));
}

int ednx_gru_step(const ednx_gru *m, const int8_t *input, const int8_t *state,
                  int8_t *output, int8_t *scratch, size_t scratch_bytes,
                  ednx_gru_trace *trace) {
    if (!m || m->magic != GRU_MAGIC || !input || !state || !output || !scratch ||
        scratch_bytes < m->hidden_size) return -1;
    uint32_t h = m->hidden_size, in = m->input_size;
    if (trace && ((!trace->codes || trace->codes_bytes < 6u * h) ||
                  (!trace->candidate_accumulators || trace->candidate_count < h))) return -1;
    for (uint32_t j = 0; j < h; ++j) {
        int32_t a[3], b[3];
        for (uint32_t gate = 0; gate < 3; ++gate) {
            uint32_t row = gate * h + j;
            a[gate] = affine(input, m->wi + row * in, m->bi[row], in, m->input_exponent + m->ei[row] + 12);
            b[gate] = affine(state, m->wh + row * h, m->bh[row], h, m->eh[row] + 5);
        }
        int8_t rl = gru_logit(m, a[0] + b[0]);
        int8_t zl = gru_logit(m, a[1] + b[1]);
        int8_t r = m->sigmoid[(int)rl + 128], z = m->sigmoid[(int)zl + 128];
        int32_t candidate = (int32_t)(a[2] + divide255(((int64_t)r + 128) * b[2]));
        int8_t nl = gru_logit(m, candidate), n = m->tanh[(int)nl + 128];
        int32_t probability = (int32_t)z + 128;
        scratch[j] = clip8(divide255((255 - probability) * (int32_t)n + probability * state[j]));
        if (trace) {
            trace->codes[j] = rl; trace->codes[h + j] = zl; trace->codes[2u * h + j] = nl;
            trace->codes[3u * h + j] = r; trace->codes[4u * h + j] = z; trace->codes[5u * h + j] = n;
            trace->candidate_accumulators[j] = candidate;
        }
    }
    memcpy(output, scratch, h);
    return 0;
}

size_t ednx_layer_norm_handle_bytes(void) { return sizeof(ednx_layer_norm); }

int ednx_layer_norm_init(ednx_layer_norm *model, const ednx_layer_norm_spec *s) {
    if (!model) return -1;
    memset(model, 0, sizeof(*model));
    if (!s || !s->size || s->size > 1024 || !s->epsilon_code ||
        s->input_exponent < -16 || s->input_exponent > 8 ||
        s->gamma_exponent < -16 || s->gamma_exponent > 8 ||
        s->output_exponent < -16 || s->output_exponent > 8 ||
        s->variance_fractional_bits > 24 || s->variance_fractional_bits % 2 ||
        !buffer_is(s->gamma, s->size) || !buffer_is(s->beta, s->size * sizeof(int32_t)) ||
        !aligned_i32(s->beta)) return -1;
    uint64_t size = s->size;
    uint64_t variance = (size * size / 4u) * 65025u;
    uint64_t squared = variance << s->variance_fractional_bits;
    if (s->epsilon_code > MAX_SQUARED_DENOMINATOR - squared) return -1;
    squared += s->epsilon_code;
    int shift = (int)(s->variance_fractional_bits / 2u) + s->gamma_exponent - s->output_exponent;
    uint64_t denominator = integer_sqrt(squared) << (shift < 0 ? -shift : 0);
    const int8_t *gamma = s->gamma.data;
    const int32_t *beta = s->beta.data;
    uint64_t gamma_peak = 0, beta_peak = 0;
    for (uint32_t j = 0; j < s->size; ++j) {
        uint64_t g = magnitude(gamma[j]), b = magnitude(beta[j]);
        if (g > gamma_peak) gamma_peak = g;
        if (b > beta_peak) beta_peak = b;
    }
    uint64_t numerator = 255u * (size - 1u) * gamma_peak;
    unsigned left = shift > 0 ? (unsigned)shift : 0;
    if (numerator > ((uint64_t)INT64_MAX >> left)) return -1;
    numerator <<= left;
    if (beta_peak > ((uint64_t)INT64_MAX - numerator) / denominator) return -1;
    numerator += beta_peak * denominator;
    if (numerator > (uint64_t)INT64_MAX - denominator / 2u) return -1;
    model->size = s->size; model->bits = s->variance_fractional_bits;
    model->shift = shift; model->epsilon = s->epsilon_code;
    model->gamma = gamma; model->beta = beta; model->magic = LN_MAGIC;
    return 0;
}

int ednx_layer_norm_frame(const ednx_layer_norm *m, const int8_t *input, int8_t *output) {
    if (!m || m->magic != LN_MAGIC || !input || !output) return -1;
    int32_t total = 0, squares = 0;
    for (uint32_t j = 0; j < m->size; ++j) {
        int32_t x = input[j]; total += x; squares += x * x;
    }
    uint64_t variance = (uint64_t)((int64_t)m->size * squares - (int64_t)total * total);
    uint64_t denominator = integer_sqrt((variance << m->bits) + m->epsilon);
    denominator <<= m->shift < 0 ? -m->shift : 0;
    uint64_t reciprocal = UINT64_MAX / denominator;
    int64_t multiplier = INT64_C(1) << (m->shift > 0 ? m->shift : 0);
    for (uint32_t j = 0; j < m->size; ++j) {
        int64_t center = (int64_t)m->size * input[j] - total;
        int64_t numerator = center * m->gamma[j] * multiplier + m->beta[j] * (int64_t)denominator;
        output[j] = clip8(round_divide(numerator, denominator, reciprocal));
    }
    return 0;
}

#ifdef EDNX_TESTING
int ednx_test_round_divide(int64_t n, uint64_t d, int64_t *output) {
    if (!d || d > INT64_MAX || !output || magnitude(n) > (uint64_t)INT64_MAX - d / 2u) return -1;
    *output = round_divide(n, d, UINT64_MAX / d);
    return 0;
}
uint64_t ednx_test_isqrt(uint64_t value) { return integer_sqrt(value); }
#endif
