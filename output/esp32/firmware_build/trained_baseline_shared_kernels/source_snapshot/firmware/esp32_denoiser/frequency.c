#include "frequency.h"
#include "integer_kernels.h"
#include <limits.h>
#include <string.h>

#define HEADER_BYTES 64u
#define LAYER_BYTES 32u
#define MAX_HISTORY_BYTES 65536u
#define MAX_WORKSPACE_BYTES (200u * 1024u)

static uint16_t u16(const uint8_t *p) { return (uint16_t)(p[0] | (uint16_t)p[1] << 8); }
static uint32_t u32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
static int32_t i32(const uint8_t *p) {
    uint32_t v = u32(p); return v <= INT32_MAX ? (int32_t)v : -1-(int32_t)~v;
}
static void put32(uint8_t *p, uint32_t v) {
    p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); p[2]=(uint8_t)(v>>16); p[3]=(uint8_t)(v>>24);
}
static int all_zero(const uint8_t *p, unsigned n) {
    unsigned i; for (i=0;i<n;++i) if(p[i]) return 0; return 1;
}
static const uint8_t *layer(const ednf_model *m, unsigned index) {
    return m->data + HEADER_BYTES + index*LAYER_BYTES;
}
static int shape(const ednf_model *m, unsigned index, unsigned dw, unsigned kt,
                 unsigned kf, unsigned stride, unsigned in, unsigned out,
                 unsigned padding, int temporal) {
    const uint8_t *l=layer(m,index);
    /* groups==in==out==1 is also represented as depthwise by the exporter. */
    if(in==1 && out==1) dw=1;
    return l[0]==dw && l[1]==kt && l[2]==kf && l[3]==stride &&
           u16(l+4)==in && u16(l+6)==out && u16(l+10)==padding &&
           (temporal ? u16(l+8)>0 : u16(l+8)==1);
}
static int topology_valid(const ednf_model *m) {
    unsigned c1=m->channels[0],c2=m->channels[1],c3=m->channels[2],g=m->global_width;
    unsigned i=0,b;
#define EXPECT(dw,kt,kf,s,in,out,p,t) do { if(!shape(m,i++,dw,kt,kf,s,in,out,p,t)) return 0; } while(0)
    EXPECT(0,1,5,2,3,c1,2,0);
    EXPECT(1,1,5,2,c1,c1,2,0); EXPECT(0,1,1,1,c1,c2,0,0);
    EXPECT(1,1,5,2,c2,c2,2,0); EXPECT(0,1,1,1,c2,c3,0,0);
    for(b=0;b<m->local_blocks;++b) {
        EXPECT(1,3,3,1,c3,c3,1,1); EXPECT(0,1,1,1,c3,c3,0,0);
    }
    EXPECT(0,1,1,1,c3*33,g,0,0);
    for(b=0;b<m->global_blocks;++b) {
        EXPECT(1,3,1,1,g,g,0,1); EXPECT(0,1,1,1,g,g,0,0);
    }
    EXPECT(0,1,1,1,g,c3*33,0,0);
    EXPECT(0,1,1,1,c3,c2,0,0); EXPECT(1,1,5,1,c2,c2,2,0);
    EXPECT(0,1,1,1,c2,c1,0,0); EXPECT(1,1,5,1,c1,c1,2,0);
    EXPECT(1,1,5,1,c1,c1,2,0); EXPECT(0,1,1,1,c1,2,0,0);
#undef EXPECT
    return i==m->layers;
}
size_t ednf_model_handle_bytes(void) { return sizeof(ednf_model); }
size_t ednf_workspace_bytes(const ednf_model *m) { return m && m->data ? m->workspace_bytes : 0; }

int ednf_init(ednf_model *m, const void *source, size_t length) {
    const uint8_t *data=(const uint8_t *)source;
    ednf_model p;
    uint64_t history=0,workspace,plane,other;
    size_t dsp,cursor;
    unsigned i;
    if(!m || !data || length<HEADER_BYTES || length>99000 ||
       memcmp(data,"EDNFQ8\0\0",8) || u32(data+8)!=1 || u32(data+28)!=length ||
       data[23] || data[27] || !all_zero(data+40,24)) return -1;
    memset(&p,0,sizeof(p)); p.data=data; p.model_bytes=length;
    for(i=0;i<3;++i) p.channels[i]=u16(data+12+2*i);
    p.global_width=u16(data+18); p.local_blocks=data[20]; p.global_blocks=data[21]; p.layers=data[22];
    p.input_exponent=(int8_t)data[24]; p.hidden_exponent=(int8_t)data[25]; p.output_exponent=(int8_t)data[26];
    if(!p.channels[0] || !p.channels[1] || !p.channels[2] || !p.global_width ||
       !p.local_blocks || !p.global_blocks || p.layers!=13u+2u*(p.local_blocks+p.global_blocks) ||
       p.input_exponent < -16 || p.input_exponent>8 || p.hidden_exponent < -12 ||
       p.hidden_exponent>0 || p.output_exponent!=-7) return -1;
    dsp=u32(data+32); cursor=HEADER_BYTES+LAYER_BYTES*p.layers;
    if(cursor>dsp || dsp>length || length-dsp!=2064u || !topology_valid(&p)) return -1;
    for(i=0;i<p.layers;++i) {
        const uint8_t *l=layer(&p,i);
        uint64_t terms=(uint64_t)(l[0]?1:u16(l+4))*l[1]*l[2];
        uint64_t elements=terms*u16(l+6);
        size_t wo=u32(l+16),bo=u32(l+20),eo=u32(l+24);
        unsigned row;
        if(!all_zero(l+14,2) || !all_zero(l+28,4) || wo%16 || bo%4 ||
           wo<cursor || wo>dsp || elements+16>dsp-wo || bo<wo+elements+16 ||
           bo>dsp || (uint64_t)bo+4u*u16(l+6)!=eo || eo>dsp || u16(l+6)>dsp-eo ||
           (int8_t)l[12]!=(i==0?p.input_exponent:p.hidden_exponent) ||
           (int8_t)l[13]!=(i==p.layers-1?p.output_exponent:p.hidden_exponent)) return -1;
        for(row=0;row<u16(l+6);++row) {
            int exponent=(int8_t)data[eo+row];
            int64_t bias=i32(data+bo+4u*row);
            uint64_t bound=(uint64_t)(bias<0?-bias:bias),k;
            for(k=0;k<terms;++k) {
                int weight=(int8_t)data[wo+row*terms+k];
                bound+=(uint64_t)128*(weight<0?-weight:weight);
            }
            if(exponent < -24 || exponent>16 || bound>INT32_MAX) return -1;
        }
        if(l[1]==3) history+=(uint64_t)2*u16(l+8)*u16(l+4)*(l[2]==3?33:1);
        cursor=eo+u16(l+6);
    }
    if(history>MAX_HISTORY_BYTES || history!=u32(data+36)) return -1;
    if(memcmp(data+dsp,"FDS1",4) || u16(data+dsp+4)!=16000 || u16(data+dsp+6)!=512 ||
       u16(data+dsp+8)!=256 || !all_zero(data+dsp+10,2) ||
       !u32(data+dsp+12) || u32(data+dsp+12)>0x41000000u) return -1;
    for(i=0;i<512;++i) if((u32(data+dsp+16+4u*i)&0x7f800000u)==0x7f800000u) return -1;
    plane=(uint64_t)p.channels[0]*257;
    other=(uint64_t)p.channels[1]*129; if(other>plane) plane=other;
    other=(uint64_t)p.channels[2]*65; if(other>plane) plane=other;
    if(p.global_width>plane) plane=p.global_width;
    workspace=4u*(p.local_blocks+p.global_blocks)+history+
              (uint64_t)p.channels[0]*129+(uint64_t)p.channels[1]*65+3*plane+p.global_width;
    if(workspace>MAX_WORKSPACE_BYTES) return -1;
    p.history_bytes=(size_t)history; p.plane_bytes=(size_t)plane; p.workspace_bytes=(size_t)workspace;
    p.dsp_data=data+dsp; *m=p; return 0;
}
int ednf_reset(const ednf_model *m, void *state, size_t length) {
    if(!m || !m->data || !state || length<m->workspace_bytes) return -1;
    memset(state,0,m->workspace_bytes); return 0;
}
static int8_t sat8(int64_t v) { return v < -128 ? -128 : v>127 ? 127 : (int8_t)v; }
static int8_t requant(int32_t value,int shift) {
    int64_t magnitude;
    if(shift>=0) { if(shift>=7) return value<0?-128:value>0?127:0; return sat8((int64_t)value*((int64_t)1<<shift)); }
    if(shift < -32) return 0;
    magnitude=value<0?-(int64_t)value:value;
    magnitude=(magnitude+((int64_t)1<<(-shift-1)))>>-shift;
    return sat8(value<0?-magnitude:magnitude);
}
static void bounded(const ednf_model *m,int8_t *x,size_t count,int relu) {
    int maximum=6*(1<<-m->hidden_exponent),minimum=relu?0:-maximum;
    size_t i; if(maximum>127) maximum=127; if(minimum < -128) minimum=-128;
    for(i=0;i<count;++i) x[i]=x[i]<minimum?(int8_t)minimum:x[i]>maximum?(int8_t)maximum:x[i];
}
static void add(const ednf_model *m,int8_t *x,const int8_t *y,size_t count) {
    size_t i; for(i=0;i<count;++i) x[i]=sat8((int)x[i]+y[i]); bounded(m,x,count,0);
}
static void repeat(const int8_t *x,int8_t *y,unsigned channels,unsigned input_f,unsigned output_f) {
    unsigned c,f; for(c=0;c<channels;++c) for(f=0;f<output_f;++f) y[c*output_f+f]=x[c*input_f+f/2];
}
/* Channel-major tensors deliberately preserve the global flatten order. */
static int conv(const ednf_model *m,unsigned index,const int8_t *x,unsigned input_f,int8_t *out,
                uint8_t *positions,unsigned *position_index,int8_t *history,size_t *history_offset) {
    const uint8_t *l=layer(m,index);
    const int8_t *w=(const int8_t *)(m->data+u32(l+16)),*exps=(const int8_t *)(m->data+u32(l+24));
    const uint8_t *bias=m->data+u32(l+20);
    unsigned in=u16(l+4),oc=u16(l+6),d=u16(l+8),kt=l[1],kf=l[2],stride=l[3],pad=u16(l+10);
    unsigned output_f=(input_f+2*pad-kf)/stride+1,c,f,t,k,ic,pos=0;
    size_t plane=(size_t)in*input_f,terms=(size_t)(l[0]?1:in)*kt*kf;
    int8_t *fifo=history+*history_offset;
    /* Gather one frequency position once, then reuse it across output rows.
     * This also covers the 1056-input global projection of the default graph. */
    if(!l[0] && kt==1 && terms<=1056) {
        int8_t packed[1056+32] __attribute__((aligned(16)));
        memset(packed+terms,0,32);
        for(f=0;f<output_f;++f) {
            for(ic=0;ic<in;++ic) for(k=0;k<kf;++k) {
                int source=(int)(f*stride+k)-(int)pad;
                packed[ic*kf+k]=source<0 || (unsigned)source>=input_f?0:x[(size_t)ic*input_f+(unsigned)source];
            }
            for(c=0;c<oc;++c) {
                int32_t acc=i32(bias+4u*c)+edn_dot_product(packed,w+c*terms,(unsigned)terms,m->data,m->model_bytes);
                out[(size_t)c*output_f+f]=requant(acc,(int8_t)l[12]+exps[c]-(int8_t)l[13]);
            }
        }
        return 0;
    }
    if(kt==3) { pos=u32(positions+4u*(*position_index)); if(pos>=2*d) return -1; }
    for(c=0;c<oc;++c) for(f=0;f<output_f;++f) {
        int32_t acc=i32(bias+4u*c);
        for(t=0;t<kt;++t) {
            const int8_t *values=kt==1 || t==2?x:fifo+(size_t)(t==0?pos:(pos+d)%(2*d))*plane;
            for(k=0;k<kf;++k) {
                int source=(int)(f*stride+k)-(int)pad;
                if(source<0 || (unsigned)source>=input_f) continue;
                if(l[0]) acc+=(int32_t)values[(size_t)c*input_f+(unsigned)source]*w[c*terms+t*kf+k];
                else for(ic=0;ic<in;++ic) acc+=(int32_t)values[(size_t)ic*input_f+(unsigned)source]*w[c*terms+((size_t)ic*kt+t)*kf+k];
            }
        }
        out[(size_t)c*output_f+f]=requant(acc,(int8_t)l[12]+exps[c]-(int8_t)l[13]);
    }
    if(kt==3) {
        memcpy(fifo+(size_t)pos*plane,x,plane);
        put32(positions+4u*(*position_index),(pos+1)%(2*d));
        ++*position_index; *history_offset+=(size_t)2*d*plane;
    }
    return 0;
}
int ednf_process_frame(const ednf_model *m,void *state,size_t length,const int8_t *features,int8_t *out) {
    uint8_t *positions=(uint8_t *)state;
    int8_t *history,*skip1,*skip2,*x,*y,*z,*global_residual;
    unsigned c1,c2,c3,g,cursor=5,b,pi=0;
    size_t ho=0;
    if(!m || !m->data || !state || length<m->workspace_bytes || !features || !out) return -1;
    c1=m->channels[0]; c2=m->channels[1]; c3=m->channels[2]; g=m->global_width;
    history=(int8_t *)state+4u*(m->local_blocks+m->global_blocks);
    skip1=history+m->history_bytes; skip2=skip1+c1*129;
    x=skip2+c2*65; y=x+m->plane_bytes; z=y+m->plane_bytes; global_residual=z+m->plane_bytes;
#define CONV(i,a,f,o) do { if(conv(m,i,a,f,o,positions,&pi,history,&ho)) return -1; } while(0)
    CONV(0,features,257,skip1); bounded(m,skip1,c1*129,0);
    CONV(1,skip1,129,y); bounded(m,y,c1*65,1);
    CONV(2,y,65,skip2); bounded(m,skip2,c2*65,0);
    CONV(3,skip2,65,y); bounded(m,y,c2*33,1);
    CONV(4,y,33,x); bounded(m,x,c3*33,0);
    for(b=0;b<m->local_blocks;++b) {
        CONV(cursor,x,33,y); bounded(m,y,c3*33,1);
        CONV(cursor+1,y,33,z); add(m,x,z,c3*33); cursor+=2;
    }
    CONV(cursor,x,1,y); bounded(m,y,g,0); ++cursor;
    for(b=0;b<m->global_blocks;++b) {
        CONV(cursor,y,1,z); bounded(m,z,g,1);
        CONV(cursor+1,z,1,global_residual); add(m,y,global_residual,g); cursor+=2;
    }
    CONV(cursor,y,1,z); add(m,x,z,c3*33); ++cursor;
    repeat(x,y,c3,33,65); CONV(cursor,y,65,z); bounded(m,z,c2*65,1);
    CONV(cursor+1,z,65,x); add(m,x,skip2,c2*65); cursor+=2;
    repeat(x,y,c2,65,129); CONV(cursor,y,129,z); bounded(m,z,c1*129,1);
    CONV(cursor+1,z,129,x); add(m,x,skip1,c1*129); cursor+=2;
    repeat(x,y,c1,129,257); CONV(cursor,y,257,z); bounded(m,z,c1*257,1);
    CONV(cursor+1,z,257,out);
#undef CONV
    return ho==m->history_bytes && pi==m->local_blocks+m->global_blocks?0:-1;
}
