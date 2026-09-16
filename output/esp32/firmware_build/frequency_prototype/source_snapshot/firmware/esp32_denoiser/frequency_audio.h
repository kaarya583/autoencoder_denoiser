#ifndef EDNF_AUDIO_H
#define EDNF_AUDIO_H
#include "frequency.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const ednf_model *model;
    void *neural_state;
    size_t neural_state_length;
    float mask_scale;
    float analysis[256], synthesis[256], synthesis_weight[256];
    float fft[1024] __attribute__((aligned(16)));
    float window[512];
    int8_t features[771], deltas[514];
} ednf_audio_state;

/* Caller provides a 16-byte-aligned audio state; no per-frame allocation.
 * Like the baseline, one initial overlap hop is discarded and a final zero hop
 * flushes the stream. FFT tables may be allocated once during initialization. */
size_t ednf_audio_state_bytes(void);
int ednf_audio_init(ednf_audio_state *state, const ednf_model *model,
                    void *neural_state, size_t neural_state_length);
int ednf_audio_reset(ednf_audio_state *state);
int ednf_audio_process(ednf_audio_state *state, const float input[256], float output[256]);
int ednf_audio_process_pcm16(ednf_audio_state *state, const int16_t input[256], int16_t output[256]);
#ifdef __cplusplus
}
#endif
#endif
