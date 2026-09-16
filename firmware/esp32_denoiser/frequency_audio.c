#include "frequency_audio.h"
#include "audio_dsp.h"

size_t ednf_audio_state_bytes(void) { return sizeof(ednf_audio_state); }
int ednf_audio_reset(ednf_audio_state *s) {
    if(!s || !s->model) return -1;
    memset(s->analysis,0,sizeof(s->analysis));
    memset(s->synthesis,0,sizeof(s->synthesis));
    memset(s->synthesis_weight,0,sizeof(s->synthesis_weight));
    return ednf_reset(s->model,s->neural_state,s->neural_state_length);
}
int ednf_audio_init(ednf_audio_state *s,const ednf_model *m,void *neural,size_t length) {
    unsigned i;
    if(!s || ((uintptr_t)s & 15u) || !m || !m->dsp_data || !neural || length<m->workspace_bytes) return -1;
    memset(s,0,sizeof(*s));
    s->model=m; s->neural_state=neural; s->neural_state_length=length;
    s->mask_scale=edn_dsp_read_f32(m->dsp_data+12);
    if(!isfinite(s->mask_scale) || s->mask_scale<=0 || s->mask_scale>8) return -1;
    for(i=0;i<512;++i) {
        s->window[i]=edn_dsp_read_f32(m->dsp_data+16+4*i);
        if(!isfinite(s->window[i]) || s->window[i]<0 || s->window[i]>1.00001f) return -1;
    }
    if(edn_dsp_init()) return -1;
    return ednf_audio_reset(s);
}
int ednf_audio_process(ednf_audio_state *s,const float input[256],float output[256]) {
    float inverse_scale;
    unsigned i;
    if(!s || !s->model || !input || !output) return -1;
    if(edn_dsp_analyze(s->analysis,s->window,s->fft,input,&inverse_scale)) return -1;
    for(i=0;i<257;++i) {
        float real=s->fft[2*i]*inverse_scale,imag=s->fft[2*i+1]*inverse_scale;
        float root=sqrtf(fmaxf(hypotf(real,imag),1e-8f));
        s->features[i]=edn_dsp_quantize_feature(root,s->model->input_exponent);
        s->features[257+i]=edn_dsp_quantize_feature(real/root,s->model->input_exponent);
        s->features[514+i]=edn_dsp_quantize_feature(imag/root,s->model->input_exponent);
    }
    if(ednf_process_frame(s->model,s->neural_state,s->neural_state_length,s->features,s->deltas)) return -1;
    return edn_dsp_synthesize(s->fft,s->window,s->synthesis,s->synthesis_weight,
                              s->deltas,s->model->output_exponent,s->mask_scale,output);
}
int ednf_audio_process_pcm16(ednf_audio_state *s,const int16_t input[256],int16_t output[256]) {
    float in[256],out[256];
    unsigned i;
    if(!input || !output) return -1;
    for(i=0;i<256;++i) in[i]=input[i]/32768.0f;
    if(ednf_audio_process(s,in,out)) return -1;
    for(i=0;i<256;++i) {
        float v=out[i]*32768.0f;
        output[i]=v < -32768 ? -32768 : v>32767 ? 32767 : (int16_t)lrintf(v);
    }
    return 0;
}
