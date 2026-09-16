#ifndef EDN_AUDIO_DSP_H
#define EDN_AUDIO_DSP_H
/* Shared conventional floating-point DSP, outside both integer neural graphs. */
#include <math.h>
#include <stdint.h>
#include <string.h>
#ifdef ESP_PLATFORM
#include "dsps_fft2r.h"
#endif

static inline float edn_dsp_read_f32(const uint8_t *p) {
    uint32_t bits = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                    ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static inline int edn_dsp_fft_forward(float *data) {
#ifdef ESP_PLATFORM
    if (dsps_fft2r_fc32(data, 512) != ESP_OK) return -1;
    return dsps_bit_rev_fc32(data, 512) == ESP_OK ? 0 : -1;
#else
    /* The host fallback verifies complete DSP behavior without ESP-IDF. */
    unsigned i, j = 0, length;
    for (i = 1; i < 512; ++i) {
        unsigned bit = 256;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) {
            float real = data[2*i], imag = data[2*i+1];
            data[2*i] = data[2*j]; data[2*i+1] = data[2*j+1];
            data[2*j] = real; data[2*j+1] = imag;
        }
    }
    for (length = 2; length <= 512; length *= 2) {
        unsigned start;
        for (start = 0; start < 512; start += length) {
            unsigned k;
            for (k = 0; k < length/2; ++k) {
                float angle = -6.2831853071795864769f * k / length;
                float wr = cosf(angle), wi = sinf(angle);
                unsigned even = 2*(start+k), odd = 2*(start+k+length/2);
                float tr = wr*data[odd] - wi*data[odd+1];
                float ti = wr*data[odd+1] + wi*data[odd];
                float er = data[even], ei = data[even+1];
                data[even] = er+tr; data[even+1] = ei+ti;
                data[odd] = er-tr; data[odd+1] = ei-ti;
            }
        }
    }
    return 0;
#endif
}

static inline int edn_dsp_init(void) {
#ifdef ESP_PLATFORM
    if (!dsps_fft2r_initialized && dsps_fft2r_init_fc32(NULL, 512) != ESP_OK) return -1;
    if (dsps_fft_w_table_size < 512) return -1;
#endif
    return 0;
}

static inline int8_t edn_dsp_quantize_feature(float value, int exponent) {
    float scaled = ldexpf(value, -exponent);
    if (scaled <= -128) return -128;
    if (scaled >= 127) return 127;
    return (int8_t)(scaled < 0 ? -floorf(-scaled + 0.5f) : floorf(scaled + 0.5f));
}

static inline int edn_dsp_analyze(float *analysis, const float *window, float *fft,
                                   const float *input, float *inverse_scale) {
    unsigned i;
    float energy=0;
    for (i = 0; i < 256; ++i) {
        float previous = analysis[i], current = input[i];
        if (!isfinite(current)) return -1;
        energy += previous*previous + current*current;
        fft[2*i] = previous*window[i];
        fft[2*i+1] = 0;
        fft[2*(i+256)] = current*window[i+256];
        fft[2*(i+256)+1] = 0;
    }
    if (!isfinite(energy)) return -1;
    *inverse_scale = 1.0f / (512.0f * sqrtf(fmaxf(energy / 512.0f, 1e-8f)));
    memcpy(analysis, input, 256 * sizeof(float));
    if (edn_dsp_fft_forward(fft)) return -1;
    return 0;
}

static inline int edn_dsp_synthesize(float *fft, const float *window, float *synthesis,
                                     float *synthesis_weight, const int8_t *deltas,
                                     int output_exponent, float mask_scale, float *output) {
    unsigned i;
    for (i = 0; i < 257; ++i) {
        float delta_r = ldexpf((float)deltas[i], output_exponent);
        float delta_i = ldexpf((float)deltas[257+i], output_exponent);
        float gain_r = 1+mask_scale*delta_r, gain_i = mask_scale*delta_i;
        float real = fft[2*i], imag = fft[2*i+1];
        fft[2*i] = real*gain_r - imag*gain_i;
        fft[2*i+1] = imag*gain_r + real*gain_i;
    }
    /* torch.irfft ignores imaginary DC/Nyquist; mirror that convention. */
    fft[1] = 0; fft[513] = 0;
    for (i = 1; i < 256; ++i) {
        fft[2*(512-i)] = fft[2*i];
        fft[2*(512-i)+1] = -fft[2*i+1];
    }
    /* Inverse FFT = conjugate(FFT(conjugate(x))) / N. Only real output is used. */
    for (i = 0; i < 512; ++i) fft[2*i+1] = -fft[2*i+1];
    if (edn_dsp_fft_forward(fft)) return -1;
    for (i = 0; i < 256; ++i) {
        float weight = window[i]*window[i];
        float value = fft[2*i] * (window[i]/512.0f);
        output[i] = (synthesis[i] + value) / fmaxf(synthesis_weight[i] + weight, 1e-8f);
        synthesis[i] = fft[2*(i+256)] * (window[i+256]/512.0f);
        synthesis_weight[i] = window[i+256]*window[i+256];
    }
    return 0;
}

#endif
