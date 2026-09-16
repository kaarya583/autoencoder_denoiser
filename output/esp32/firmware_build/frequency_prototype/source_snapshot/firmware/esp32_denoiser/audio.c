#include "audio.h"

#include "audio_dsp.h"


int edn_audio_reset(edn_audio_state *s) {
    if (!s || !s->model) return -1;
    memset(s->analysis, 0, sizeof(s->analysis));
    memset(s->synthesis, 0, sizeof(s->synthesis));
    memset(s->synthesis_weight, 0, sizeof(s->synthesis_weight));
    return edn_reset(s->model, s->neural_state, s->neural_state_length);
}

int edn_audio_init(edn_audio_state *s, const edn_model *m, void *neural_state, size_t state_length) {
    const uint8_t *p;
    unsigned i;
    if (!s || !m || !m->dsp_data || m->dsp_bytes != 3988 ||
        m->input_channels != 387 || m->output_channels != 514 ||
        !neural_state || state_length < m->state_bytes) return -1;
    memset(s, 0, sizeof(*s));
    s->model = m; s->neural_state = neural_state; s->neural_state_length = state_length;
    s->mask_scale = edn_dsp_read_f32(m->dsp_data + 16);
    if (!isfinite(s->mask_scale) || s->mask_scale <= 0 || s->mask_scale > 8) return -1;
    p = m->dsp_data + 20;
    for (i = 0; i < 512; ++i) {
        s->window[i] = edn_dsp_read_f32(p + 4*i);
        if (!isfinite(s->window[i]) || s->window[i] < 0 || s->window[i] > 1.00001f) return -1;
    }
    p += 2048;
    memcpy(s->lower, p, 192); p += 192;
    memcpy(s->upper, p, 192); p += 192;
    for (i = 0; i < 192; ++i) {
        s->lower_weight[i] = edn_dsp_read_f32(p + 4*i);
        s->upper_weight[i] = edn_dsp_read_f32(p + 768 + 4*i);
        if (s->lower[i] >= 64 || s->upper[i] >= 64 ||
            !isfinite(s->lower_weight[i]) || !isfinite(s->upper_weight[i]) ||
            s->lower_weight[i] < 0 || s->upper_weight[i] < 0 ||
            s->lower_weight[i] > 1.00001f || s->upper_weight[i] > 1.00001f) return -1;
    }
    if (edn_dsp_init()) return -1;
    return edn_audio_reset(s);
}


int edn_audio_process(edn_audio_state *s, const float input[256], float output[256]) {
    unsigned i;
    float inverse_scale;
    if (!s || !s->model || !input || !output) return -1;
    if (edn_dsp_analyze(s->analysis, s->window, s->fft, input, &inverse_scale)) return -1;
    memset(s->bands, 0, sizeof(s->bands));
    for (i = 0; i < 257; ++i) {
        float real = s->fft[2*i]*inverse_scale, imag = s->fft[2*i+1]*inverse_scale;
        float magnitude = hypotf(real, imag);
        if (i < 65) {
            s->bands[0][i] = magnitude; s->bands[1][i] = real; s->bands[2][i] = imag;
        } else {
            unsigned high = i-65, lower = 65+s->lower[high], upper = 65+s->upper[high];
            float a = s->lower_weight[high], b = s->upper_weight[high];
            s->bands[0][lower] += magnitude*a; s->bands[0][upper] += magnitude*b;
            s->bands[1][lower] += real*a; s->bands[1][upper] += real*b;
            s->bands[2][lower] += imag*a; s->bands[2][upper] += imag*b;
        }
    }
    for (i = 0; i < 129; ++i) {
        float root = sqrtf(fmaxf(s->bands[0][i], 1e-8f));
        s->features[i] = edn_dsp_quantize_feature(root, s->model->input_exponent);
        s->features[129+i] = edn_dsp_quantize_feature(s->bands[1][i]/root, s->model->input_exponent);
        s->features[258+i] = edn_dsp_quantize_feature(s->bands[2][i]/root, s->model->input_exponent);
    }
    if (edn_process_frame(s->model, s->neural_state, s->neural_state_length, s->features, s->deltas)) return -1;
    return edn_dsp_synthesize(s->fft, s->window, s->synthesis, s->synthesis_weight,
                              s->deltas, s->model->output_exponent, s->mask_scale, output);
}

int edn_audio_process_pcm16(edn_audio_state *s, const int16_t input[256], int16_t output[256]) {
    float in[256], out[256];
    unsigned i;
    if (!input || !output) return -1;
    for (i = 0; i < 256; ++i) in[i] = input[i] / 32768.0f;
    if (edn_audio_process(s, in, out)) return -1;
    for (i = 0; i < 256; ++i) {
        float value = out[i]*32768.0f;
        output[i] = value < -32768 ? -32768 : value > 32767 ? 32767 : (int16_t)lrintf(value);
    }
    return 0;
}
