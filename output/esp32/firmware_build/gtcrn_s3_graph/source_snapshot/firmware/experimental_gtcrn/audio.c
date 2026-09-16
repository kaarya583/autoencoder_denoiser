#include "audio.h"
#include "internal.h"
#include "../experimental_dsp/gtcrn_erb.h"
#include "../esp32_denoiser/audio_dsp.h"

#define AUDIO_MAGIC UINT32_C(0x47415531)
struct edng_audio {
    uint32_t magic;
    const edng_model *model;
    int8_t *neural;
    void *workspace;
    ednx_erb *erb;
    size_t neural_bytes, workspace_bytes, audio_bytes;
    const float *window;
    int normalize, input_exponent;
    float rms_floor_squared;
    float analysis[256], synthesis[256], synthesis_weight[256];
    float fft[1024] __attribute__((aligned(16)));
    /* After ERB analysis, reuse these arrays for expanded/compressed masks. */
    float spectra[3*257], bands[3*129];
    int8_t features[387], mask[258];
};

static size_t align8(size_t value) {return (value+7u)&~(size_t)7u;}
size_t edng_audio_bytes(void) {return (align8(sizeof(edng_audio))+ednx_erb_handle_bytes()+15u)&~(size_t)15u;}
static int overlaps(const void *left,size_t lb,const void *right,size_t rb) {
    uintptr_t a=(uintptr_t)left,b=(uintptr_t)right;
    if(a>UINTPTR_MAX-lb||b>UINTPTR_MAX-rb) return 1;
    return a<b+rb&&b<a+lb;
}
static int valid(const edng_audio *s) {return s&&s->magic==AUDIO_MAGIC&&s->model&&s->model->magic==EDNG_MAGIC;}
static int io_valid(const edng_audio *s,const void *input,const void *output,size_t bytes) {
    if(!valid(s)||!input||!output) return 0;
    const void *regions[]={s,s->model,s->model->packed,s->neural,s->workspace};
    size_t sizes[]={s->audio_bytes,edng_handle_bytes(),s->model->packed_bytes,s->neural_bytes,s->workspace_bytes};
    for(unsigned i=0;i<5;++i)
        if(overlaps(input,bytes,regions[i],sizes[i])||overlaps(output,bytes,regions[i],sizes[i])) return 0;
    return 1;
}

int edng_audio_reset(edng_audio *s) {
    if(!valid(s)) return -1;
    memset(s->analysis,0,sizeof(s->analysis));
    memset(s->synthesis,0,sizeof(s->synthesis));
    memset(s->synthesis_weight,0,sizeof(s->synthesis_weight));
    return edng_reset(s->model,s->neural,s->neural_bytes);
}

int edng_audio_init(edng_audio *s,size_t bytes,const edng_model *model,
                    int8_t *neural,size_t neural_bytes,void *workspace,size_t workspace_bytes) {
    int exponent;
    if(!s||((uintptr_t)s&15u)||bytes<sizeof(edng_audio)||(uintptr_t)s>UINTPTR_MAX-bytes) return -1;
    /* Preserve invalid aliased caller storage; otherwise a failed reinit must
     * invalidate the old audio state before any ordinary argument rejection. */
    if((model&&overlaps(s,bytes,model,edng_handle_bytes()))||
       (neural&&overlaps(s,bytes,neural,neural_bytes))||
       (workspace&&overlaps(s,bytes,workspace,workspace_bytes))||
       (model&&model->magic==EDNG_MAGIC&&overlaps(s,bytes,model->packed,model->packed_bytes))) return -1;
    memset(s,0,sizeof(*s));
    if(bytes<edng_audio_bytes()||!model||
       edng_input_exponent(model,&exponent)||!neural||neural_bytes<edng_state_bytes()||
       !workspace||workspace_bytes<edng_workspace_bytes()) return -1;
    const void *regions[]={s,model,model->packed,neural,workspace};
    size_t sizes[]={bytes,edng_handle_bytes(),model->packed_bytes,neural_bytes,workspace_bytes};
    for(unsigned i=0;i<5;++i) for(unsigned j=0;j<i;++j)
        if(overlaps(regions[i],sizes[i],regions[j],sizes[j])) return -1;
    memset(s,0,edng_audio_bytes());
    s->model=model;s->neural=neural;s->neural_bytes=neural_bytes;s->workspace=workspace;s->workspace_bytes=workspace_bytes;s->audio_bytes=bytes;
    s->window=(const float *)model->operators[EDNG_OP_WINDOW].array;
    s->normalize=model->packed[20];s->input_exponent=exponent;
    /* Packed parser already checked LE host and IEEE754 DSP metadata. */
    double floor;memcpy(&floor,model->packed+32,sizeof(floor));
    s->rms_floor_squared=(float)(floor*floor);
    s->erb=(ednx_erb *)((uint8_t *)s+align8(sizeof(edng_audio)));
    if(ednx_erb_init(s->erb,model->operators[EDNG_OP_ERB].array,2040)||edn_dsp_init()) return -1;
    s->magic=AUDIO_MAGIC;
    return edng_audio_reset(s);
}

int edng_audio_process(edng_audio *s,const float input[256],float output[256]) {
    if(!io_valid(s,input,output,256*sizeof(float))||((uintptr_t)input%sizeof(float))||((uintptr_t)output%sizeof(float))) return -1;
    float energy=0;
    for(unsigned i=0;i<256;++i) {
        float previous=s->analysis[i],current=input[i];
        if(!isfinite(current)) return -1;
        if(s->normalize) energy+=previous*previous+current*current;
        s->fft[2*i]=previous*s->window[i];s->fft[2*i+1]=0;
        s->fft[2*(i+256)]=current*s->window[i+256];s->fft[2*(i+256)+1]=0;
    }
    if(!isfinite(energy)) return -1;
    memcpy(s->analysis,input,256*sizeof(float));
    if(edn_dsp_fft_forward(s->fft)) return -1;
    float scale=s->normalize?512.0f*sqrtf(fmaxf(energy/512.0f,s->rms_floor_squared)):1.0f;
    if(!isfinite(scale)||scale<=0) return -1;
    for(unsigned i=0;i<257;++i) {
        float real=s->fft[2*i]/scale,imag=s->fft[2*i+1]/scale;
        s->spectra[i]=sqrtf(real*real+imag*imag+1e-12f);
        s->spectra[257+i]=real;s->spectra[514+i]=imag;
    }
    if(ednx_erb_forward(s->erb,s->spectra,3u*257u,s->bands,3u*129u,3)) return -1;
    for(unsigned i=0;i<387;++i) s->features[i]=edn_dsp_quantize_feature(s->bands[i],s->input_exponent);
    if(edng_process_frame(s->model,s->neural,s->neural_bytes,s->features,387,s->mask,258,s->workspace,s->workspace_bytes)) return -1;
    for(unsigned i=0;i<258;++i) s->bands[i]=(float)s->mask[i]/128.0f;
    if(ednx_erb_inverse(s->erb,s->bands,2u*129u,s->spectra,2u*257u,2)) return -1;
    for(unsigned i=0;i<257;++i) {
        /* Applying the mask to the original spectrum is algebraically equal
         * to restoring frame RMS after masking the normalized spectrum. */
        float real=s->fft[2*i],imag=s->fft[2*i+1],mr=s->spectra[i],mi=s->spectra[257+i];
        s->fft[2*i]=real*mr-imag*mi;s->fft[2*i+1]=imag*mr+real*mi;
    }
    /* Real inverse FFT discards imaginary DC and Nyquist. */
    s->fft[1]=0;s->fft[513]=0;
    for(unsigned i=1;i<256;++i) {s->fft[2*(512-i)]=s->fft[2*i];s->fft[2*(512-i)+1]=-s->fft[2*i+1];}
    for(unsigned i=0;i<512;++i) s->fft[2*i+1]=-s->fft[2*i+1];
    if(edn_dsp_fft_forward(s->fft)) return -1;
    for(unsigned i=0;i<256;++i) {
        float weight=s->window[i]*s->window[i],value=s->fft[2*i]*(s->window[i]/512.0f);
        output[i]=(s->synthesis[i]+value)/fmaxf(s->synthesis_weight[i]+weight,1e-8f);
        if(!isfinite(output[i])) return -1;
        s->synthesis[i]=s->fft[2*(i+256)]*(s->window[i+256]/512.0f);
        s->synthesis_weight[i]=s->window[i+256]*s->window[i+256];
    }
    return 0;
}

int edng_audio_process_pcm16(edng_audio *s,const int16_t input[256],int16_t output[256]) {
    if(!io_valid(s,input,output,256*sizeof(int16_t))||((uintptr_t)input%sizeof(int16_t))||((uintptr_t)output%sizeof(int16_t))) return -1;
    float in[256],out[256];
    for(unsigned i=0;i<256;++i) in[i]=input[i]/32768.0f;
    if(edng_audio_process(s,in,out)) return -1;
    for(unsigned i=0;i<256;++i) {
        float value=out[i]*32768.0f;
        output[i]=value < -32768?-32768:value > 32767?32767:(int16_t)lrintf(value);
    }
    return 0;
}
