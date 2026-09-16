#include "operators.h"
#include <limits.h>
#include <string.h>

#define EDNX_AFFINE_MAGIC UINT32_C(0x58414631)
#define EDNX_DIM_LIMIT UINT32_C(65536)
struct ednx_affine { uint32_t magic; ednx_affine_spec s; size_t row_weights; };

static int grid(int e) { return e >= -16 && e <= 8; }
static int dimension(uint32_t n) { return n > 0 && n <= EDNX_DIM_LIMIT; }
static int product_size(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX/a) return -1;
    *out = a*b; return 0;
}
static int tensor_size(uint32_t c, uint32_t t, uint32_t f, size_t *out) {
    size_t n;
    return product_size(c,t,&n) || product_size(n,f,out);
}
static int8_t clipped(int64_t x) { return (int8_t)(x < -128 ? -128 : (x > 127 ? 127 : x)); }
static int64_t shifted(int64_t n, int shift) {
    uint64_t magnitude,q;
    if (shift >= 0) return n*(INT64_C(1)<<shift);
    magnitude=n < 0 ? (uint64_t)(-n) : (uint64_t)n;
    q=(magnitude+(UINT64_C(1)<<(-shift-1)))>>-shift;
    return n < 0 ? -(int64_t)q : (int64_t)q;
}
/* Same portable limb identity independently tested in primitives.c. Keeping
 * the scalar helper private allows either isolated source to link alone. */
static uint64_t multiply_high(uint64_t a,uint64_t b) {
    uint64_t a0=(uint32_t)a,a1=a>>32,b0=(uint32_t)b,b1=b>>32;
    uint64_t low=a0*b0,middle=a1*b0+(low>>32),carry=middle>>32;
    middle=(uint32_t)middle+a0*b1;
    return a1*b1+carry+(middle>>32);
}
typedef struct { int64_t multiplier; uint64_t denominator,reciprocal; } rational_scale;
static rational_scale prepare_rational(int shift,uint64_t denominator) {
    rational_scale scale;
    scale.multiplier=shift >= 0 ? INT64_C(1)<<shift : 1;
    scale.denominator=denominator*(shift < 0 ? UINT64_C(1)<<-shift : 1);
    /* One division per vector operation, never per frequency product. */
    scale.reciprocal=UINT64_MAX/scale.denominator;
    return scale;
}
static int8_t rational(int64_t n,const rational_scale *scale) {
    uint64_t magnitude,q;
    n*=scale->multiplier;
    magnitude=(n < 0 ? (uint64_t)(-n) : (uint64_t)n)+scale->denominator/2;
    q=multiply_high(magnitude,scale->reciprocal);
    if (magnitude-q*scale->denominator >= scale->denominator) ++q;
    return clipped(n < 0 ? -(int64_t)q : (int64_t)q);
}
static int valid(const ednx_affine *m) { return m && m->magic == EDNX_AFFINE_MAGIC; }

size_t ednx_affine_handle_bytes(void) { return sizeof(ednx_affine); }
int ednx_affine_init(ednx_affine *m, const ednx_affine_spec *s) {
    size_t row, weight_bytes, bias_bytes;
    uint32_t o;
    ednx_affine p;
    if (!m) return -1;
    memset(m,0,sizeof(*m));
    if (!s || s->kind > EDNX_CONV_TRANSPOSE2D || !dimension(s->input_channels) ||
        !dimension(s->output_channels) || !dimension(s->groups) ||
        s->input_channels%s->groups || s->output_channels%s->groups ||
        !grid(s->input_exponent) || !grid(s->output_exponent)) return -1;
    if (!dimension(s->kernel_time) || !dimension(s->kernel_frequency) ||
        s->kernel_time > 1024 || s->kernel_frequency > 1024 ||
        !dimension(s->stride_time) || !dimension(s->stride_frequency) ||
        !dimension(s->dilation_time) || !dimension(s->dilation_frequency) ||
        s->padding_time > EDNX_DIM_LIMIT || s->padding_frequency > EDNX_DIM_LIMIT) return -1;
    if (s->kind == EDNX_LINEAR && (s->groups != 1 || s->kernel_time != 1 || s->kernel_frequency != 1 ||
        s->stride_time != 1 || s->stride_frequency != 1 || s->padding_time || s->padding_frequency ||
        s->dilation_time != 1 || s->dilation_frequency != 1)) return -1;
    if (s->kind != EDNX_CONV_TRANSPOSE2D) {
        if (s->output_padding_time || s->output_padding_frequency) return -1;
    } else if ((s->output_padding_time >= s->stride_time && s->output_padding_time >= s->dilation_time) ||
               (s->output_padding_frequency >= s->stride_frequency && s->output_padding_frequency >= s->dilation_frequency)) return -1;
    if (tensor_size(s->input_channels/s->groups,s->kernel_time,s->kernel_frequency,&row) ||
        product_size(row,s->output_channels,&weight_bytes) ||
        product_size(s->output_channels,sizeof(int32_t),&bias_bytes)) return -1;
    if (!s->weights.data || s->weights.bytes != weight_bytes || !s->bias.data || s->bias.bytes != bias_bytes ||
        (uintptr_t)s->bias.data%sizeof(int32_t) || !s->exponents.data || s->exponents.bytes != s->output_channels) return -1;
    for (o=0;o<s->output_channels;++o) {
        const int8_t *weights = (const int8_t*)s->weights.data+o*row;
        int64_t bias = ((const int32_t*)s->bias.data)[o];
        uint64_t bound = (uint64_t)(bias < 0 ? -bias : bias);
        int exponent = ((const int8_t*)s->exponents.data)[o];
        size_t i;
        if (exponent < -20 || exponent > 4) return -1;
        for (i=0;i<row;++i) {
            int w = weights[i];
            bound += (uint64_t)(128*(w < 0 ? -w : w));
            if (bound > INT32_MAX) return -1;
        }
        if (bound > INT32_MAX) return -1;
    }
    p.magic=EDNX_AFFINE_MAGIC; p.s=*s; p.row_weights=row; *m=p;
    return 0;
}

int ednx_affine_output_shape(const ednx_affine *m, uint32_t t, uint32_t f, uint32_t *ot, uint32_t *of) {
    int64_t a,b;
    const ednx_affine_spec *s;
    if (!valid(m) || !dimension(t) || !dimension(f) || !ot || !of) return -1;
    s=&m->s;
    if (s->kind == EDNX_LINEAR) { if (f != 1) return -1; *ot=t; *of=1; return 0; }
    if (s->kind == EDNX_CONV2D) {
        a=(int64_t)t+2*s->padding_time-(int64_t)s->dilation_time*(s->kernel_time-1)-1;
        b=(int64_t)f+2*s->padding_frequency-(int64_t)s->dilation_frequency*(s->kernel_frequency-1)-1;
        if (a < 0 || b < 0) return -1;
        a=a/s->stride_time+1; b=b/s->stride_frequency+1;
    } else {
        a=(int64_t)(t-1)*s->stride_time-2*s->padding_time+(int64_t)s->dilation_time*(s->kernel_time-1)+s->output_padding_time+1;
        b=(int64_t)(f-1)*s->stride_frequency-2*s->padding_frequency+(int64_t)s->dilation_frequency*(s->kernel_frequency-1)+s->output_padding_frequency+1;
    }
    if (a < 1 || a > EDNX_DIM_LIMIT || b < 1 || b > EDNX_DIM_LIMIT) return -1;
    *ot=(uint32_t)a; *of=(uint32_t)b; return 0;
}

int ednx_affine_run(const ednx_affine *m, const int8_t *x, size_t xb, uint32_t t, uint32_t f,
                     int8_t *y, size_t yb, int32_t *trace, size_t trace_count) {
    uint32_t ot,of,o,at,af;
    size_t input_count,output_count;
    const ednx_affine_spec *s;
    if (ednx_affine_output_shape(m,t,f,&ot,&of)) return -1;
    s=&m->s;
    if (tensor_size(s->input_channels,t,f,&input_count) || tensor_size(s->output_channels,ot,of,&output_count) ||
        !x || !y || xb < input_count || yb < output_count ||
        (trace && (trace_count < output_count || (uintptr_t)trace%sizeof(int32_t)))) return -1;
    for (o=0;o<s->output_channels;++o) {
        uint32_t inputs=s->input_channels/s->groups;
        uint32_t first=(o/(s->output_channels/s->groups))*inputs;
        const int8_t *w=(const int8_t*)s->weights.data+o*m->row_weights;
        int shift=s->input_exponent+((const int8_t*)s->exponents.data)[o]-s->output_exponent;
        for (at=0;at<ot;++at) for (af=0;af<of;++af) {
            int32_t sum=((const int32_t*)s->bias.data)[o];
            uint32_t i,kt,kf;
            size_t out;
            if (s->kind == EDNX_LINEAR) {
                for (i=0;i<inputs;++i) sum+=(int32_t)x[(size_t)at*inputs+i]*w[i];
                out=(size_t)at*s->output_channels+o;
            } else {
                for (i=0;i<inputs;++i) for (kt=0;kt<s->kernel_time;++kt) for (kf=0;kf<s->kernel_frequency;++kf) {
                    int64_t it,jf;
                    if (s->kind == EDNX_CONV2D) {
                        it=(int64_t)at*s->stride_time-s->padding_time+(int64_t)kt*s->dilation_time;
                        jf=(int64_t)af*s->stride_frequency-s->padding_frequency+(int64_t)kf*s->dilation_frequency;
                    } else {
                        it=(int64_t)at+s->padding_time-(int64_t)kt*s->dilation_time;
                        jf=(int64_t)af+s->padding_frequency-(int64_t)kf*s->dilation_frequency;
                        /* Positive inverse coordinates are at most output
                         * size + padding <=131072; use native-width divides. */
                        if (it < 0 || jf < 0 || (uint32_t)it%s->stride_time || (uint32_t)jf%s->stride_frequency) continue;
                        it=(uint32_t)it/s->stride_time; jf=(uint32_t)jf/s->stride_frequency;
                    }
                    if (it >= 0 && it < t && jf >= 0 && jf < f) {
                        size_t ix=((size_t)(first+i)*t+(size_t)it)*f+(size_t)jf;
                        size_t wi=((size_t)i*s->kernel_time+kt)*s->kernel_frequency+kf;
                        sum+=(int32_t)x[ix]*w[wi];
                    }
                }
                out=((size_t)o*ot+at)*of+af;
            }
            y[out]=clipped(shifted(sum,shift));
            if (trace) trace[out]=sum;
        }
    }
    return 0;
}

static int stream_shape(const ednx_affine *m,uint32_t f,uint32_t stride,int left,int right,
                         uint32_t *history,uint32_t *transformed) {
    int64_t expanded,kept,width,h;
    if (!valid(m) || m->s.kind != EDNX_CONV2D || m->s.stride_time != 1 || m->s.padding_time ||
        !dimension(f) || !dimension(stride) || stride > 1024 || left < -65536 || left > 65536 || right < -65536 || right > 65536) return -1;
    expanded=(int64_t)f*stride;
    kept=expanded+(left < 0 ? left : 0)+(right < 0 ? right : 0);
    width=expanded+left+right;
    h=(int64_t)(m->s.kernel_time-1)*m->s.dilation_time;
    if (kept < 1 || width < 1 || width > EDNX_DIM_LIMIT || h >= EDNX_DIM_LIMIT) return -1;
    *history=(uint32_t)h; *transformed=(uint32_t)width; return 0;
}
size_t ednx_stream_history_bytes(const ednx_affine *m,uint32_t f) {
    uint32_t h,pf; size_t count;
    if (stream_shape(m,f,1,0,0,&h,&pf) || tensor_size(m->s.input_channels,h,f,&count)) return 0;
    return count;
}
size_t ednx_stream_workspace_bytes(const ednx_affine *m,uint32_t f,uint32_t stride,int left,int right) {
    uint32_t h,pf; size_t count;
    if (stream_shape(m,f,stride,left,right,&h,&pf) || tensor_size(m->s.input_channels,h+1,pf,&count)) return 0;
    return count;
}
int ednx_stream_conv(const ednx_affine *m,const int8_t *frame,size_t fb,const int8_t *history,size_t hb,
                      uint32_t f,uint32_t stride,int left,int right,int8_t *out,size_t ob,
                      int8_t *next,size_t nb,int8_t *work,size_t wb) {
    uint32_t h,pf,c,t,j,ot,of; size_t required,states,frames,outs;
    if (stream_shape(m,f,stride,left,right,&h,&pf) ||
        tensor_size(m->s.input_channels,h+1,pf,&required) || tensor_size(m->s.input_channels,h,f,&states) ||
        product_size(m->s.input_channels,f,&frames) || ednx_affine_output_shape(m,h+1,pf,&ot,&of) ||
        tensor_size(m->s.output_channels,ot,of,&outs) || ot != 1 ||
        !frame || !out || !work || fb < frames || ob < outs || wb < required ||
        (states && (!history || !next || hb < states || nb < states))) return -1;
    for (c=0;c<m->s.input_channels;++c) for (t=0;t<=h;++t) for (j=0;j<pf;++j) {
        int64_t source=(int64_t)j-left;
        int8_t value=0;
        if (source >= 0 && source < (int64_t)f*stride && (uint32_t)source%stride == 0) {
            size_t fi=(uint32_t)source/stride;
            value=t == h ? frame[(size_t)c*f+fi] : history[((size_t)c*h+t)*f+fi];
        }
        work[((size_t)c*(h+1)+t)*pf+j]=value;
    }
    if (ednx_affine_run(m,work,required,h+1,pf,out,ob,NULL,0)) return -1;
    if (h) for (c=0;c<m->s.input_channels;++c) {
        if (h > 1) memmove(next+(size_t)c*h*f,history+((size_t)c*h+1)*f,(size_t)(h-1)*f);
        memcpy(next+((size_t)c*h+h-1)*f,frame+(size_t)c*f,f);
    }
    return 0;
}

int ednx_regrid(const int8_t *x,int8_t *y,size_t n,int ie,int oe) {
    size_t i;
    if (!x || !y || !n || !grid(ie) || !grid(oe)) return -1;
    for (i=0;i<n;++i) y[i]=clipped(shifted(x[i],ie-oe));
    return 0;
}
int ednx_residual(const int8_t *a,const int8_t *b,int8_t *y,size_t n,int ae,int be,int oe) {
    size_t i; int fine=ae < be ? ae : be;
    if (!a || !b || !y || !n || !grid(ae) || !grid(be) || !grid(oe)) return -1;
    if (oe < fine) fine=oe;
    for (i=0;i<n;++i) y[i]=clipped(shifted(shifted(a[i],ae-fine)+shifted(b[i],be-fine),fine-oe));
    return 0;
}
int ednx_prelu(const int8_t *x,int8_t *y,uint32_t channels,size_t inner,const int8_t *slopes,size_t count,int se,int ie,int oe) {
    uint32_t c; size_t i,n;
    if (!x || !y || !slopes || !dimension(channels) || !inner || product_size(channels,inner,&n) ||
        (count != 1 && count != channels) || se < -20 || se > 4 || !grid(ie) || !grid(oe)) return -1;
    for (c=0;c<channels;++c) for (i=0;i<inner;++i) {
        size_t k=(size_t)c*inner+i;
        int value=x[k];
        y[k]=clipped(value >= 0 ? shifted(value,ie-oe) : shifted((int64_t)value*slopes[count == 1 ? 0 : c],ie+se-oe));
    }
    return 0;
}
int ednx_lut(const int8_t *x,int8_t *y,size_t n,const int8_t table[256]) {
    size_t i;
    if (!x || !y || !n || !table) return -1;
    for (i=0;i<n;++i) y[i]=table[(int)x[i]+128];
    return 0;
}
int ednx_attention_energy(const int8_t *x,int8_t *y,size_t rows,uint32_t f,int ie,int oe) {
    size_t row,n; uint32_t j; rational_scale scale;
    if (!x || !y || !rows || !f || f > 1024 || product_size(rows,f,&n) || !grid(ie) || !grid(oe)) return -1;
    scale=prepare_rational(2*ie-oe,f);
    for (row=0;row<rows;++row) {
        int32_t sum=0;
        for (j=0;j<f;++j) { int32_t value=x[row*f+j]; sum+=value*value; }
        y[row]=rational(sum,&scale);
    }
    return 0;
}
int ednx_attention_product(const int8_t *x,const int8_t *p,int8_t *y,size_t rows,uint32_t f,int ie,int oe) {
    size_t row,n; uint32_t j; rational_scale scale;
    if (!x || !p || !y || !rows || !dimension(f) || product_size(rows,f,&n) || !grid(ie) || !grid(oe)) return -1;
    scale=prepare_rational(ie-oe,255);
    for (row=0;row<rows;++row) for (j=0;j<f;++j) y[row*f+j]=rational((int64_t)x[row*f+j]*((int)p[row]+128),&scale);
    return 0;
}
int ednx_subband(const int8_t *x,int8_t *y,uint32_t c,uint32_t t,uint32_t f) {
    uint32_t channel,time,j,neighbor; size_t n,out;
    if (!x || !y || !dimension(c) || !dimension(t) || !dimension(f) || tensor_size(c,t,f,&n) || product_size(n,3,&out)) return -1;
    for (channel=0;channel<c;++channel) for (neighbor=0;neighbor<3;++neighbor) for (time=0;time<t;++time) for (j=0;j<f;++j) {
        int64_t source=(int64_t)j+neighbor-1;
        y[(((size_t)channel*3+neighbor)*t+time)*f+j]=source < 0 || source >= f ? 0 : x[((size_t)channel*t+time)*f+(size_t)source];
    }
    return 0;
}
int ednx_shuffle(const int8_t *a,const int8_t *b,int8_t *y,uint32_t c,size_t inner) {
    uint32_t channel; size_t n,out;
    if (!a || !b || !y || !dimension(c) || !inner || product_size(c,inner,&n) || product_size(n,2,&out)) return -1;
    for (channel=0;channel<c;++channel) {
        memcpy(y+(size_t)(2*channel)*inner,a+(size_t)channel*inner,inner);
        memcpy(y+(size_t)(2*channel+1)*inner,b+(size_t)channel*inner,inner);
    }
    return 0;
}
