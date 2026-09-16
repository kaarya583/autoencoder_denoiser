#ifndef EDN_AUDIO_H
#define EDN_AUDIO_H

#include "denoiser.h"

#define EDN_FFT_SIZE 512
#define EDN_HOP_SIZE 256

typedef struct {
    const edn_model *model;
    void *neural_state;
    size_t neural_state_length;
    float mask_scale;
    float analysis[256];
    float synthesis[256];
    float synthesis_weight[256];
    /* ESP32-S3's optimized FFT requires 16-byte input alignment. */
    float fft[1024] __attribute__((aligned(16)));
    float window[512];
    float bands[3][129];
    float lower_weight[192];
    float upper_weight[192];
    uint8_t lower[192];
    uint8_t upper[192];
    int8_t features[387];
    int8_t deltas[514];
    /* Keep this scalar in the existing tail padding of the 32-bit S3 layout. */
    float output_gain;
} edn_audio_state;

/* Initialize FFT tables and decode the DSP constants stored in the model.
 * ESP builds use ESP-DSP. Host builds use a portable radix-two FFT for testing.
 * Init can allocate vendor FFT tables; processing allocates no memory.
 */
int edn_audio_init(edn_audio_state *state, const edn_model *model,
                   void *neural_state, size_t neural_state_length);
int edn_audio_reset(edn_audio_state *state);
int edn_audio_process(edn_audio_state *state, const float input[256], float output[256]);
int edn_audio_process_pcm16(edn_audio_state *state, const int16_t input[256], int16_t output[256]);

/* One initial emitted hop is overlap padding. Feed one zero hop at stream end
 * to flush the last real output. Preserve state across all normal hops.
 */
#endif
