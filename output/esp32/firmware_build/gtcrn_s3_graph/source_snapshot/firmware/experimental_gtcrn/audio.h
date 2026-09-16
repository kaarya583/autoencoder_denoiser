#ifndef EDNG_AUDIO_H
#define EDNG_AUDIO_H

/* Complete causal waveform frontend around the integer GTCRN neural graph.
 * FFT, magnitude/ERB, optional frame RMS, complex masking and OLA are ordinary
 * float32 DSP. All learned operations and persistent neural histories remain
 * integer. ESP builds use the shared ESP-DSP FFT; host builds use radix-two C.
 * Caller owns all buffers. Audio state requires16-byte alignment; model,
 * neural state and workspace obey graph.h. These regions must be disjoint.
 * Initialization may initialize vendor FFT tables; processing has no heap.
 * Failed initialization invalidates a safely writable, nonoverlapping state;
 * invalid or aliased state storage is left unchanged.
 */
#include "graph.h"

typedef struct edng_audio edng_audio;
size_t edng_audio_bytes(void);
int edng_audio_init(edng_audio *audio, size_t audio_bytes, const edng_model *model,
                    int8_t *neural_state, size_t neural_bytes,
                    void *workspace, size_t workspace_bytes);
int edng_audio_reset(edng_audio *audio);
/* Exactly256 samples per call, preserving state between hops. Input/output
 * may overlap each other, but must be disjoint from all model/state/workspace
 * regions. On a processing error, discard/reset audio state before reuse.
 * Discard the initial emitted overlap hop; feed one zero hop at stream end.
 */
int edng_audio_process(edng_audio *audio, const float input[256], float output[256]);
int edng_audio_process_pcm16(edng_audio *audio, const int16_t input[256], int16_t output[256]);

#endif
