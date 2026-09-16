#include "operators.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
    ednx_affine_spec s={0};
    int8_t weights[72], exps[4]={-4,-5,-6,-7}; int32_t bias[4]={12,-24,36,-48};
    int8_t input[4*5*33],output[4*8*66],a[4096],b[4096],y[12288],slopes[4]={-128,0,64,127};
    int8_t history[4*4*33]={0},workspace[4*5*70]; uint32_t ot,of; int e,o; size_t i;
    ednx_affine *model=(ednx_affine*)malloc(ednx_affine_handle_bytes());
    for(i=0;i<sizeof(weights);++i)weights[i]=(int8_t)((int)(i%255)-127);
    for(i=0;i<sizeof(input);++i)input[i]=(int8_t)i;
    for(i=0;i<sizeof(a);++i){a[i]=(int8_t)i;b[i]=(int8_t)(255-i);}
    s.kind=EDNX_CONV2D;s.input_channels=4;s.output_channels=4;s.groups=2;
    s.kernel_time=3;s.kernel_frequency=3;s.stride_time=1;s.stride_frequency=1;
    s.padding_frequency=1;s.dilation_time=2;s.dilation_frequency=1;
    s.input_exponent=-4;s.output_exponent=-2;
    s.weights=(ednx_buffer){weights,sizeof(weights)};s.bias=(ednx_buffer){bias,sizeof(bias)};s.exponents=(ednx_buffer){exps,sizeof(exps)};
    assert(ednx_affine_init(model,&s)==0);
    assert(ednx_affine_run(model,input,sizeof(input),5,33,output,sizeof(output),0,0)==0);
    assert(ednx_stream_conv(model,input,4*33,history,sizeof(history),33,2,1,0,output,sizeof(output),history,sizeof(history),workspace,sizeof(workspace))==0);
    assert(ednx_stream_conv(model,input,4*33,history,sizeof(history),33,2,-2,-3,output,sizeof(output),history,sizeof(history),workspace,sizeof(workspace))==0);
    s.kind=EDNX_CONV_TRANSPOSE2D;s.stride_time=2;s.stride_frequency=2;s.output_padding_time=1;
    assert(ednx_affine_init(model,&s)==0);
    assert(ednx_affine_output_shape(model,2,17,&ot,&of)==0);
    assert(ednx_affine_run(model,input,sizeof(input),2,17,output,sizeof(output),0,0)==0);
    for(e=-16;e<=8;++e)for(o=-16;o<=8;++o){
      assert(ednx_regrid(a,y,4096,e,o)==0);
      assert(ednx_residual(a,b,y,4096,e,o,-16)==0);
      assert(ednx_prelu(a,y,4,1024,slopes,4,4,e,o)==0);
      assert(ednx_prelu(a,y,4,1024,slopes,4,-20,e,o)==0);
      assert(ednx_attention_energy(a,y,4,1024,e,o)==0);
      assert(ednx_attention_product(a,b,y,4,1024,e,o)==0);
    }
    assert(ednx_subband(a,y,4,1,1024)==0);assert(ednx_shuffle(a,b,y,4,1024)==0);
    s.kind=EDNX_LINEAR;s.input_channels=s.output_channels=s.groups=1;s.kernel_time=s.kernel_frequency=1;
    s.stride_time=s.stride_frequency=s.dilation_time=s.dilation_frequency=1;
    s.padding_time=s.padding_frequency=s.output_padding_time=s.output_padding_frequency=0;
    weights[0]=0;exps[0]=4;bias[0]=2147483647;s.input_exponent=8;s.output_exponent=-16;
    s.weights.bytes=1;s.exponents.bytes=1;s.bias.bytes=4;
    assert(ednx_affine_init(model,&s)==0);assert(ednx_affine_run(model,a,1,1,1,y,1,0,0)==0);assert(y[0]==127);
    bias[0]=-2147483647;
    assert(ednx_affine_init(model,&s)==0);assert(ednx_affine_run(model,a,1,1,1,y,1,0,0)==0);assert(y[0]==-128);
    free(model);return 0;
}
