#ifndef EDN_DENOISER_H
#define EDN_DENOISER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The neural runtime uses integer arithmetic only. Audio transforms are external.
 * Model memory must outlive this handle; state is caller-owned and allocation-free.
 */
typedef struct {
    const uint8_t *data;
    size_t model_bytes;
    size_t state_bytes;
    uint16_t input_channels;
    uint16_t hidden_channels;
    uint16_t output_channels;
    uint16_t blocks;
    int8_t input_exponent;
    int8_t hidden_exponent;
    int8_t output_exponent;
    const uint8_t *dsp_data;
    size_t dsp_bytes;
} edn_model;

/* Runs signed, extreme-value, alignment, maximum-length, and tail known-answer
 * cases against the scalar dot. Execute before timing on the intended board. */
const char *edn_neural_backend(void);
int edn_backend_self_test(void);

/* Return 0 on success, -1 for an invalid model or argument. */
int edn_init(edn_model *model, const void *data, size_t length);
int edn_reset(const edn_model *model, void *state, size_t state_length);
int edn_process_frame(const edn_model *model, void *state, size_t state_length,
                      const int8_t *features, int8_t *complex_mask_delta);

#ifdef __cplusplus
}
#endif
#endif
