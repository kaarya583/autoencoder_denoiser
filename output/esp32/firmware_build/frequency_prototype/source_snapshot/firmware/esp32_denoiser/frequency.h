#ifndef EDNF_FREQUENCY_H
#define EDNF_FREQUENCY_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* EDNFQ8-v1 fixed frequency U-Net. Model memory and workspace are caller-owned. */
typedef struct {
    const uint8_t *data, *dsp_data;
    size_t model_bytes, workspace_bytes, history_bytes, plane_bytes;
    uint16_t channels[3], global_width;
    uint8_t local_blocks, global_blocks, layers;
    int8_t input_exponent, hidden_exponent, output_exponent;
} ednf_model;
size_t ednf_model_handle_bytes(void);
size_t ednf_workspace_bytes(const ednf_model *model);
int ednf_init(ednf_model *model, const void *data, size_t length);
int ednf_reset(const ednf_model *model, void *workspace, size_t length);
/* Contiguous channel-major INT8 features [3,257], output deltas [2,257]. */
int ednf_process_frame(const ednf_model *model, void *workspace, size_t length,
                       const int8_t *features, int8_t *deltas);
#ifdef __cplusplus
}
#endif
#endif
