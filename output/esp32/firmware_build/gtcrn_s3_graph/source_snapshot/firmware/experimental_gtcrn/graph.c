#include "internal.h"
#include <string.h>

/* Largest activation is SFE's9*129 codes. The stream buffer covers the
 * largest decoder history after explicit frequency padding:16*11*35. */
#define ACTIVATION_BYTES 1168u
typedef struct {
    int8_t main[2][ACTIVATION_BYTES];
    int8_t skips[3152];
    int8_t temporary[4][ACTIVATION_BYTES];
    int8_t regrid[ACTIVATION_BYTES];
    int8_t stream[6160];
    int8_t gru_input[16], gru_hidden[16], gru_scratch[16];
} edng_workspace;

typedef struct { const int8_t *data; size_t count; int exponent; } edge;
typedef struct {
    uint16_t point1, point_act, depth, depth_act, point2, attention, attention_fc, sigmoid;
    uint16_t energy_input, energy_output, product_input, product_output, output;
} gt_block;

#define OP(name) EDNG_OP_##name
#define GRID(name) EDNG_GRID_##name
#define GT(name) {OP(name##_POINT_CONV1),OP(name##_POINT_ACT),OP(name##_DEPTH_CONV), \
                 OP(name##_DEPTH_ACT),OP(name##_POINT_CONV2),OP(name##_TRA_ATT_GRU_0), \
                 OP(name##_TRA_ATT_FC),OP(name##_TRA_ATT_ACT), \
                 GRID(name##_TRA_ENERGY_INPUT),GRID(name##_TRA_ENERGY_OUTPUT), \
                 GRID(name##_TRA_PRODUCT_INPUT),GRID(name##_TRA_PRODUCT_OUTPUT),GRID(name##_OUTPUT)}
static const gt_block blocks[6] = {
    GT(ENCODER_EN_CONVS_2), GT(ENCODER_EN_CONVS_3), GT(ENCODER_EN_CONVS_4),
    GT(DECODER_DE_CONVS_0), GT(DECODER_DE_CONVS_1), GT(DECODER_DE_CONVS_2)
};
static const size_t convolution_offsets[6] = {0, 1056, 3168, 8448, 13728, 15840};
static const size_t attention_offsets[6] = {16896, 16912, 16928, 16944, 16960, 16976};
static const size_t inter_offsets[2] = {16992, 17520};
static const size_t skip_offsets[5] = {0, 1040, 1568, 2096, 2624};
static const uint16_t skip_grids[5] = {
    GRID(DECODER_DE_CONVS_0_SKIP_ADD), GRID(DECODER_DE_CONVS_1_SKIP_ADD),
    GRID(DECODER_DE_CONVS_2_SKIP_ADD), GRID(DECODER_DE_CONVS_3_SKIP_ADD), GRID(DECODER_DE_CONVS_4_SKIP_ADD)
};

static int valid(const edng_model *m) { return m && m->magic == EDNG_MAGIC; }
static int separate(const void *a, size_t an, const void *b, size_t bn) {
    uintptr_t first = (uintptr_t)a, second = (uintptr_t)b;
    if (!a || !b || an > UINTPTR_MAX-first || bn > UINTPTR_MAX-second) return 0;
    return first+an <= second || second+bn <= first;
}

size_t edng_state_bytes(void) { return EDNG_NEURAL_STATE_BYTES; }
size_t edng_workspace_bytes(void) { return sizeof(edng_workspace); }

int edng_input_exponent(const edng_model *m, int *exponent) {
    if (!valid(m) || !exponent) return -1;
    *exponent = m->grids[GRID(ERB_OUTPUT)];
    return 0;
}

int edng_reset(const edng_model *m, int8_t *state, size_t bytes) {
    if (!valid(m) || bytes < EDNG_NEURAL_STATE_BYTES ||
        !separate(state, EDNG_NEURAL_STATE_BYTES, m, edng_handle_bytes()) ||
        !separate(state, EDNG_NEURAL_STATE_BYTES, m->packed, m->packed_bytes)) return -1;
    memset(state, 0, EDNG_NEURAL_STATE_BYTES);
    return 0;
}

static int affine(const edng_model *m, uint16_t id, edge input, uint32_t time, uint32_t frequency,
                   int8_t *output, edng_workspace *w, edge *result) {
    const edng_operator *op = &m->operators[id];
    uint32_t ot, of;
    if (input.count > ACTIVATION_BYTES ||
        ednx_regrid(input.data, w->regrid, input.count, input.exponent, op->input_exponent) ||
        ednx_affine_output_shape(op->handle, time, frequency, &ot, &of)) return -1;
    size_t count = (size_t)edng_layout[id].dimensions[1]*ot*of;
    if (count > ACTIVATION_BYTES ||
        ednx_affine_run(op->handle, w->regrid, input.count, time, frequency, output, count, NULL, 0)) return -1;
    *result = (edge){output, count, op->output_exponent};
    return 0;
}

static int activation(const edng_model *m, uint16_t id, edge input, uint32_t channels,
                       int8_t *output, edng_workspace *w, edge *result) {
    const edng_operator *op = &m->operators[id];
    if (!channels || input.count % channels || input.count > ACTIVATION_BYTES ||
        ednx_regrid(input.data, w->regrid, input.count, input.exponent, op->input_exponent)) return -1;
    int status = edng_layout[id].kind == 3
        ? ednx_prelu(w->regrid, output, channels, input.count/channels, op->array, 1, op->auxiliary,
                      op->input_exponent, op->output_exponent)
        : ednx_lut(w->regrid, output, input.count, op->array);
    if (status) return -1;
    /* Sigmoid's exponent field is deliberately meaningless; its result is
     * consumed directly by attention_product, never ordinary regridding. */
    *result = (edge){output, input.count, op->output_exponent};
    return 0;
}

static int convolution_block(const edng_model *m, uint16_t conv, uint16_t act, edge input,
                               uint32_t frequency, int8_t *output, edng_workspace *w, edge *result) {
    edge intermediate;
    if (affine(m, conv, input, 1, frequency, w->temporary[1], w, &intermediate)) return -1;
    return activation(m, act, intermediate, edng_layout[conv].dimensions[1], output, w, result);
}

static int gru_cell(const edng_model *m, uint16_t id, const int8_t *input, int exponent,
                     int8_t *state, int8_t *output, edng_workspace *w) {
    const edng_operator *op = &m->operators[id];
    size_t inputs = edng_layout[id].dimensions[0], hidden = edng_layout[id].dimensions[1];
    if (inputs > sizeof(w->gru_input) || hidden > sizeof(w->gru_scratch) ||
        ednx_regrid(input, w->gru_input, inputs, exponent, op->input_exponent) ||
        ednx_gru_step(op->handle, w->gru_input, state, output, w->gru_scratch, sizeof(w->gru_scratch), NULL)) return -1;
    if (output != state) memcpy(state, output, hidden);
    return 0;
}

static int temporal_block(const edng_model *m, unsigned index, edge input, int8_t *state,
                           int8_t *output, edng_workspace *w, edge *result) {
    const gt_block *b = &blocks[index];
    if (input.count != 528) return -1;
    edge value = {w->temporary[0], 792, input.exponent}, gate;
    if (ednx_subband(input.data, w->temporary[0], 8, 1, 33) ||
        affine(m, b->point1, value, 1, 33, w->temporary[1], w, &value) ||
        activation(m, b->point_act, value, 16, w->temporary[2], w, &value)) return -1;
    const edng_operator *depth = &m->operators[b->depth];
    size_t history = ednx_stream_history_bytes(depth->handle, 33);
    int padding = (edng_layout[b->depth].flags & 2) ? 1 : 0;
    if (history != (size_t)16*33*2*edng_layout[b->depth].dimensions[7] ||
        ednx_regrid(value.data, w->regrid, value.count, value.exponent, depth->input_exponent) ||
        ednx_stream_conv(depth->handle, w->regrid, value.count,
                          state+convolution_offsets[index], history, 33, 1, padding, padding,
                          w->temporary[1], 528, state+convolution_offsets[index], history,
                          w->stream, sizeof(w->stream))) return -1;
    value = (edge){w->temporary[1], 528, depth->output_exponent};
    if (activation(m, b->depth_act, value, 16, w->temporary[2], w, &value) ||
        affine(m, b->point2, value, 1, 33, w->temporary[1], w, &value)) return -1;
    if (ednx_regrid(value.data, w->regrid, 264, value.exponent, m->grids[b->energy_input]) ||
        ednx_attention_energy(w->regrid, w->temporary[2], 8, 33,
                                m->grids[b->energy_input], m->grids[b->energy_output]) ||
        gru_cell(m, b->attention, w->temporary[2], m->grids[b->energy_output],
                   state+attention_offsets[index], w->temporary[3], w)) return -1;
    gate = (edge){w->temporary[3], 16, -7};
    if (affine(m, b->attention_fc, gate, 1, 1, w->temporary[2], w, &gate) ||
        activation(m, b->sigmoid, gate, 8, w->temporary[3], w, &gate) ||
        ednx_regrid(value.data, w->regrid, 264, value.exponent, m->grids[b->product_input]) ||
        ednx_attention_product(w->regrid, gate.data, w->temporary[2], 8, 33,
                                 m->grids[b->product_input], m->grids[b->product_output]) ||
        ednx_regrid(w->temporary[2], w->regrid, 264, m->grids[b->product_output], m->grids[b->output]) ||
        ednx_regrid(input.data+264, w->temporary[3], 264, input.exponent, m->grids[b->output]) ||
        ednx_shuffle(w->regrid, w->temporary[3], output, 8, 33)) return -1;
    *result = (edge){output, 528, m->grids[b->output]};
    return 0;
}

/* Dual-path tensors are rearranged explicitly between channel-major [16,33]
 * convolution layout and [33,16] frequency-major recurrent/LN layout. */
static void frequency_major(const int8_t *input, int8_t *output) {
    for (size_t f=0; f<33; ++f) for (size_t c=0; c<16; ++c) output[f*16+c] = input[c*33+f];
}
static void channel_major(const int8_t *input, int8_t *output) {
    for (size_t f=0; f<33; ++f) for (size_t c=0; c<16; ++c) output[c*33+f] = input[f*16+c];
}

static int grouped_frequency(const edng_model *m, uint16_t first, edge input,
                               int8_t *output, edng_workspace *w) {
    for (unsigned group=0; group<2; ++group) for (unsigned direction=0; direction<2; ++direction) {
        uint16_t id = (uint16_t)(first + group*2 + direction);
        memset(w->gru_hidden, 0, 4);
        for (unsigned step=0; step<33; ++step) {
            unsigned f = direction ? 32-step : step;
            int8_t *target = output+f*16+group*8+direction*4;
            if (gru_cell(m, id, input.data+f*16+group*8, input.exponent, w->gru_hidden, target, w)) return -1;
        }
    }
    return 0;
}

static int grouped_temporal(const edng_model *m, uint16_t first, edge input,
                              int8_t *state, int8_t *output, edng_workspace *w) {
    for (unsigned f=0; f<33; ++f) for (unsigned group=0; group<2; ++group) {
        size_t start = f*16+group*8;
        if (gru_cell(m, (uint16_t)(first+group), input.data+start, input.exponent,
                       state+start, output+start, w)) return -1;
    }
    return 0;
}

static int normalize(const edng_model *m, uint16_t id, edge input,
                       int8_t *output, edng_workspace *w, edge *result) {
    const edng_operator *op = &m->operators[id];
    if (input.count != 528 ||
        ednx_regrid(input.data, w->regrid, 528, input.exponent, op->input_exponent) ||
        ednx_layer_norm_frame(op->handle, w->regrid, output)) return -1;
    *result = (edge){output, 528, op->output_exponent};
    return 0;
}

static int dual_path(const edng_model *m, unsigned index, edge input, int8_t *state,
                       int8_t *output, edng_workspace *w, edge *result) {
    uint16_t first = index ? OP(DPGRNN2_INTRA_RNN_RNN1_0) : OP(DPGRNN1_INTRA_RNN_RNN1_0);
    uint16_t intra_grid = index ? GRID(DPGRNN2_INTRA_ADD) : GRID(DPGRNN1_INTRA_ADD);
    uint16_t inter_grid = index ? GRID(DPGRNN2_INTER_ADD) : GRID(DPGRNN1_INTER_ADD);
    if (input.count != 528) return -1;
    frequency_major(input.data, w->temporary[0]);
    edge original = {w->temporary[0], 528, input.exponent}, value;
    if (grouped_frequency(m, first, original, w->temporary[1], w)) return -1;
    value = (edge){w->temporary[1], 528, -7};
    if (affine(m, (uint16_t)(first+4), value, 33, 1, w->temporary[2], w, &value) ||
        normalize(m, (uint16_t)(first+5), value, w->temporary[1], w, &value) ||
        ednx_residual(original.data, value.data, w->temporary[2], 528,
                        original.exponent, value.exponent, m->grids[intra_grid])) return -1;
    edge intra = {w->temporary[2], 528, m->grids[intra_grid]};
    if (grouped_temporal(m, (uint16_t)(first+6), intra, state+inter_offsets[index], w->temporary[1], w)) return -1;
    value = (edge){w->temporary[1], 528, -7};
    if (affine(m, (uint16_t)(first+8), value, 33, 1, w->temporary[0], w, &value) ||
        normalize(m, (uint16_t)(first+9), value, w->temporary[1], w, &value) ||
        ednx_residual(intra.data, value.data, w->temporary[0], 528,
                        intra.exponent, value.exponent, m->grids[inter_grid])) return -1;
    channel_major(w->temporary[0], output);
    *result = (edge){output, 528, m->grids[inter_grid]};
    return 0;
}

int edng_process_frame(const edng_model *m, int8_t *state, size_t state_bytes,
                        const int8_t *input, size_t input_bytes, int8_t *output, size_t output_bytes,
                        void *workspace, size_t workspace_bytes) {
    if (!valid(m) || state_bytes < EDNG_NEURAL_STATE_BYTES || workspace_bytes < sizeof(edng_workspace) ||
        input_bytes != 387 || output_bytes != 258) return -1;
    const void *buffers[] = {m, m->packed, state, input, output, workspace};
    const size_t lengths[] = {edng_handle_bytes(), m->packed_bytes, EDNG_NEURAL_STATE_BYTES, 387, 258, sizeof(edng_workspace)};
    for (unsigned i=0; i<6; ++i) for (unsigned j=0; j<i; ++j)
        if (!separate(buffers[i], lengths[i], buffers[j], lengths[j])) return -1;
    edng_workspace *w = workspace;
    edge x, skips[5];
    unsigned slot=0;
    if (ednx_subband(input, w->main[slot], 3, 1, 129)) return -1;
    x = (edge){w->main[slot], 1161, m->grids[GRID(ERB_OUTPUT)]};
    for (unsigned index=0; index<5; ++index) {
        int status;
        slot ^= 1;
        if (index < 2) status = convolution_block(m, (uint16_t)(index*2), (uint16_t)(index*2+1),
                                                    x, index ? 65 : 129, w->main[slot], w, &x);
        else status = temporal_block(m, index-2, x, state, w->main[slot], w, &x);
        if (status) return -1;
        memcpy(w->skips+skip_offsets[index], x.data, x.count);
        skips[index] = (edge){w->skips+skip_offsets[index], x.count, x.exponent};
    }
    for (unsigned index=0; index<2; ++index) {
        slot ^= 1;
        if (dual_path(m, index, x, state, w->main[slot], w, &x)) return -1;
    }
    for (unsigned index=0; index<5; ++index) {
        edge skip = skips[4-index];
        slot ^= 1;
        if (x.count != skip.count || ednx_residual(x.data, skip.data, w->main[slot], x.count,
                                                    x.exponent, skip.exponent, m->grids[skip_grids[index]])) return -1;
        x = (edge){w->main[slot], x.count, m->grids[skip_grids[index]]};
        slot ^= 1;
        int status = index < 3
            ? temporal_block(m, index+3, x, state, w->main[slot], w, &x)
            : convolution_block(m, index == 3 ? OP(DECODER_DE_CONVS_3_CONV) : OP(DECODER_DE_CONVS_4_CONV),
                                  index == 3 ? OP(DECODER_DE_CONVS_3_ACT) : OP(DECODER_DE_CONVS_4_ACT),
                                  x, index == 3 ? 33 : 65, w->main[slot], w, &x);
        if (status) return -1;
    }
    if (x.count != 258 || x.exponent != -7) return -1;
    memcpy(output, x.data, 258);
    return 0;
}
