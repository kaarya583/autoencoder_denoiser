/* GTI8PK01 bounded immutable loader. No heap allocation or float neural math.
 * Float reads below validate only external DSP metadata/window/ERB constants.
 * SHA256 is integrity/file association, not an authentication or lineage proof. */
#include "internal.h"
#include "packed_tables.h"
#include <float.h>
#include <math.h>
#include <string.h>

#if FLT_RADIX != 2 || FLT_MANT_DIG != 24 || DBL_MANT_DIG != 53
#error "The packed DSP metadata requires IEEE754 float32/float64"
#endif
typedef char edng_float_size[sizeof(float) == 4 ? 1 : -1];
typedef char edng_double_size[sizeof(double) == 8 ? 1 : -1];

static uint16_t u16(const uint8_t *p) { return (uint16_t)(p[0] | (uint16_t)p[1] << 8); }
static uint32_t u32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24; }
static uint64_t u64(const uint8_t *p) { return u32(p) | (uint64_t)u32(p+4)<<32; }
static int signed_byte(uint8_t value) { return value <= 127 ? (int)value : (int)value-256; }
static float f32(const uint8_t *p) { uint32_t bits=u32(p); float value; memcpy(&value,&bits,4); return value; }
static double f64(const uint8_t *p) { uint64_t bits=u64(p); double value; memcpy(&value,&bits,8); return value; }
static size_t align8(size_t value) { return (value+7u)&~(size_t)7u; }
static size_t align16(size_t value) { return (value+15u)&~(size_t)15u; }
static int zero(const uint8_t *p, size_t bytes) { for(size_t i=0;i<bytes;++i) if(p[i]) return 0; return 1; }

/* SHA256 compression and streaming state, with unsigned modular arithmetic. */
typedef struct { uint32_t h[8]; uint64_t bytes; uint8_t block[64]; size_t used; } sha256;
static uint32_t rotate(uint32_t x, unsigned n) { return (x>>n)|(x<<(32u-n)); }
static void sha_block(sha256 *s,const uint8_t *p) {
    static const uint32_t k[64]={
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t w[64];
    for(unsigned i=0;i<16;++i) w[i]=(uint32_t)p[4*i]<<24|(uint32_t)p[4*i+1]<<16|(uint32_t)p[4*i+2]<<8|p[4*i+3];
    for(unsigned i=16;i<64;++i) {
        uint32_t a=w[i-15],b=w[i-2];
        w[i]=w[i-16]+(rotate(a,7)^rotate(a,18)^(a>>3))+w[i-7]+(rotate(b,17)^rotate(b,19)^(b>>10));
    }
    uint32_t a=s->h[0],b=s->h[1],c=s->h[2],d=s->h[3],e=s->h[4],f=s->h[5],g=s->h[6],h=s->h[7];
    for(unsigned i=0;i<64;++i) {
        uint32_t t1=h+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^(~e&g))+k[i]+w[i];
        uint32_t t2=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c));
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    s->h[0]+=a;s->h[1]+=b;s->h[2]+=c;s->h[3]+=d;s->h[4]+=e;s->h[5]+=f;s->h[6]+=g;s->h[7]+=h;
}
static void sha_init(sha256 *s) {
    static const uint32_t initial[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    memcpy(s->h,initial,sizeof(initial));s->bytes=0;s->used=0;
}
static void sha_add(sha256 *s,const uint8_t *p,size_t bytes) {
    s->bytes+=bytes;
    while(bytes) {
        size_t take=64-s->used;if(take>bytes) take=bytes;
        memcpy(s->block+s->used,p,take);s->used+=take;p+=take;bytes-=take;
        if(s->used==64) { sha_block(s,s->block);s->used=0; }
    }
}
static void sha_finish(sha256 *s,uint8_t result[32]) {
    uint64_t bits=s->bytes*8u;
    s->block[s->used++]=0x80;
    if(s->used>56) { memset(s->block+s->used,0,64-s->used);sha_block(s,s->block);s->used=0; }
    memset(s->block+s->used,0,56-s->used);
    for(unsigned i=0;i<8;++i) s->block[63-i]=(uint8_t)(bits>>(8*i));
    sha_block(s,s->block);
    for(unsigned i=0;i<8;++i) for(unsigned j=0;j<4;++j) result[4*i+j]=(uint8_t)(s->h[i]>>(24-8*j));
}
static int table_hash(const uint8_t *table,const uint8_t expected[32]) {
    sha256 s;uint8_t digest[32];sha_init(&s);sha_add(&s,table,256);sha_finish(&s,digest);
    return memcmp(digest,expected,32)==0;
}
static int integrity(const uint8_t *data,size_t bytes) {
    sha256 s;uint8_t digest[32];static const uint8_t zeros[32]={0};
    sha_init(&s);sha_add(&s,data,44);sha_add(&s,zeros,32);sha_add(&s,data+76,bytes-76);sha_finish(&s,digest);
    return memcmp(digest,data+44,32)==0;
}

static size_t native_bytes(unsigned kind) {
    if(kind==1||kind==2) return ednx_affine_handle_bytes();
    if(kind==4) return ednx_gru_handle_bytes();
    if(kind==5) return ednx_layer_norm_handle_bytes();
    return 0;
}
size_t edng_handle_bytes(void) {
    size_t bytes=align8(sizeof(edng_model));
    for(unsigned i=0;i<EDNG_RECORDS;++i) bytes+=align8(native_bytes(edng_layout[i].kind));
    return bytes;
}

/* Each reference's byte length/type follows only the compiled fixed topology,
 * never unchecked dimensions from the file. Types1/2/3/4 are i8/i32/u8/f32. */
static size_t array_bytes(const edng_layout_record *r,unsigned slot,unsigned *type) {
    unsigned i=r->dimensions[0],o=r->dimensions[1];*type=1;
    if(r->kind==1||r->kind==2) {
        if(slot==0) return (size_t)o*i/r->dimensions[2]*(r->subtype==1?1:r->dimensions[3]*r->dimensions[4]);
        if(slot==1) {*type=2;return 4u*o;} return o;
    }
    if(r->kind==4) {
        if(slot==0) return 3u*o*i;
        if(slot==1) return 3u*o*o;
        if(slot==2||slot==3) {*type=2;return 12u*o;}
        if(slot==4||slot==5) return 3u*o;
        return 256;
    }
    if(r->kind==5) {if(slot) {*type=2;return 4u*528u;}return 528;}
    if(r->kind==3) return 1;
    if(r->kind==6||r->kind==7) return 256;
    if(r->kind==8) {*type=3;return 2040;}
    *type=4;return 2048;
}

/* New blocks appear in first-reference order, aliases refer to a prior block
 * with identical extent/type. Scanning at most272 fixed refs avoids an init
 * heap or multi-kilobyte temporary array directory on the MCU stack. */
static int array_valid(const uint8_t *data,size_t bytes,unsigned record,unsigned slot,size_t *cursor) {
    const edng_layout_record *r=&edng_layout[record];unsigned type;
    size_t length=array_bytes(r,slot,&type);
    uint32_t offset=u32(data+EDNG_RECORD_OFFSET+80u*record+24u+4u*slot);
    if(offset<EDNG_ARRAY_OFFSET||(offset&15u)||offset>bytes||length>bytes-offset) return 0;
    if(offset==align16(*cursor)) {
        if(!zero(data+*cursor,offset-*cursor)) return 0;
        *cursor=offset+length;return 1;
    }
    if(offset>=*cursor) return 0;
    for(unsigned previous=0;previous<=record;++previous) {
        const edng_layout_record *p=&edng_layout[previous];
        unsigned slots=previous==record?slot:p->references;
        for(unsigned j=0;j<slots;++j) {
            if(u32(data+EDNG_RECORD_OFFSET+80u*previous+24u+4u*j)==offset) {
                unsigned previous_type;size_t previous_length=array_bytes(p,j,&previous_type);
                return length==previous_length&&type==previous_type;
            }
        }
    }
    return 0;
}
static ednx_buffer buffer(const uint8_t *data,unsigned record,unsigned slot) {
    unsigned type;ednx_buffer b;
    b.data=data+u32(data+EDNG_RECORD_OFFSET+80u*record+24u+4u*slot);
    b.bytes=array_bytes(&edng_layout[record],slot,&type);return b;
}

static int erb_valid(const uint8_t *data) {
    if(u16(data)!=0||u16(data+128)!=382) return 0;
    for(unsigned row=0;row<64;++row) {
        unsigned start=u16(data+row*2),end=u16(data+(row+1)*2);
        if(end<start||end>382) return 0;
        for(unsigned i=start;i<end;++i) {
            float value=f32(data+512+4u*i);
            if(data[130+i]>=192||(i>start&&data[130+i]<=data[129+i])||!isfinite(value)||value==0) return 0;
        }
    }
    return 1;
}
static int window_valid(const uint8_t *data) {
    if(f32(data)!=0||f32(data+256u*4u)!=1) return 0;
    for(unsigned i=0;i<512;++i) {float v=f32(data+4u*i);if(!isfinite(v)||v<0||v>1) return 0;}
    return 1;
}

int edng_init(edng_model *model,size_t handle_bytes,const void *packed,size_t bytes) {
    uintptr_t mp=(uintptr_t)model,pp=(uintptr_t)packed;
    if(!model||(mp&7u)||handle_bytes<sizeof(edng_model)||mp>UINTPTR_MAX-handle_bytes) return -1;
    /* Do not clear a handle that aliases the immutable packed buffer. */
    if(packed&&((pp>=mp&&pp-mp<handle_bytes)||(mp>=pp&&mp-pp<bytes))) return -1;
    memset(model,0,sizeof(*model));
    if(!packed||(pp&7u)||bytes<192||bytes>99000||pp>UINTPTR_MAX-bytes) return -1;
    if(handle_bytes<edng_handle_bytes()) return -1;
    const uint8_t *data=packed;
    const uint16_t endian=1;
    if(*(const uint8_t *)&endian!=1) return -1; /* Native i32 array views are LE. */
    if(memcmp(data,"GTI8PK01",8)||u16(data+8)!=1||u16(data+10)!=192||u32(data+12)!=bytes||
       u32(data+16)!=1||u32(data+20)>1||u32(data+24)!=80||u16(data+28)!=EDNG_GRIDS||
       u16(data+30)!=EDNG_RECORDS||!zero(data+172,20)||bytes<EDNG_ARRAY_OFFSET||(bytes&15u)) return -1;
    if(zero(data+108,32)||zero(data+140,32)||!integrity(data,bytes)) return -1;
    double floor=f64(data+32),square=floor*floor;
    if(!isfinite(floor)||floor<=0||square<FLT_MIN||square>FLT_MAX) return -1;
    int initial_input=signed_byte(data[40]),state=signed_byte(data[41]);
    int logit=signed_byte(data[42]),accumulator=signed_byte(data[43]);
    if(initial_input < -12||initial_input>0||state!=-7||logit < -6||logit > -2||accumulator!=-12) return -1;
    for(unsigned i=0;i<EDNG_GRIDS;++i) {int value=signed_byte(data[192+i]);if(value < -16||value>8) return -1;}
    if(!zero(data+192+EDNG_GRIDS,EDNG_RECORD_OFFSET-192-EDNG_GRIDS)) return -1;
    size_t cursor=EDNG_ARRAY_OFFSET,arena=align8(sizeof(edng_model));
    for(unsigned index=0;index<EDNG_RECORDS;++index) {
        const edng_layout_record *layout=&edng_layout[index];const uint8_t *r=data+EDNG_RECORD_OFFSET+index*80u;
        if(u16(r)!=index||r[2]!=layout->kind||r[3]!=layout->flags||r[6]!=layout->subtype) return -1;
        for(unsigned j=0;j<8;++j) if(u16(r+8+2u*j)!=layout->dimensions[j]) return -1;
        int in=signed_byte(r[4]),out=signed_byte(r[5]),aux=signed_byte(r[7]);
        if(in!=(layout->input_grid<0?0:signed_byte(data[192+layout->input_grid]))||
           out!=(layout->output_grid<0?0:signed_byte(data[192+layout->output_grid]))) return -1;
        if((layout->kind!=5&&!zero(r+64,16))||(layout->kind!=3&&layout->kind!=5&&aux)) return -1;
        for(unsigned j=layout->references;j<10;++j) if(u32(r+24+4u*j)) return -1;
        for(unsigned j=0;j<layout->references;++j) if(!array_valid(data,bytes,index,j,&cursor)) return -1;
        edng_operator *op=&model->operators[index];
        op->input_exponent=in;op->output_exponent=out;op->auxiliary=aux;
        size_t need=native_bytes(layout->kind);
        if(need) {if(arena>handle_bytes||need>handle_bytes-arena) return -1;op->handle=(uint8_t *)model+arena;arena+=align8(need);}
        if(layout->kind==1||layout->kind==2) {
            ednx_affine_spec s;memset(&s,0,sizeof(s));
            s.kind=layout->subtype-1;s.input_channels=layout->dimensions[0];s.output_channels=layout->dimensions[1];s.groups=layout->dimensions[2];
            s.kernel_time=layout->dimensions[3];s.kernel_frequency=layout->dimensions[4];
            s.stride_time=1;s.stride_frequency=layout->dimensions[5];s.padding_frequency=layout->dimensions[6];
            s.dilation_time=layout->dimensions[7];s.dilation_frequency=1;s.input_exponent=in;s.output_exponent=out;
            if(s.kind==EDNX_LINEAR) {s.kernel_time=s.kernel_frequency=s.stride_frequency=s.dilation_time=1;}
            s.weights=buffer(data,index,0);s.bias=buffer(data,index,1);s.exponents=buffer(data,index,2);
            if(ednx_affine_init(op->handle,&s)) return -1;
        } else if(layout->kind==4) {
            if(in < -12||in>0||out!=-7) return -1;
            ednx_gru_spec s;s.input_size=layout->dimensions[0];s.hidden_size=layout->dimensions[1];
            s.input_exponent=in;s.state_exponent=state;s.logit_exponent=logit;s.accumulator_exponent=accumulator;
            s.weight_ih=buffer(data,index,0);s.weight_hh=buffer(data,index,1);s.bias_ih=buffer(data,index,2);s.bias_hh=buffer(data,index,3);
            s.exponent_ih=buffer(data,index,4);s.exponent_hh=buffer(data,index,5);s.sigmoid_lut=buffer(data,index,6);s.tanh_lut=buffer(data,index,7);
            if(!table_hash(s.sigmoid_lut.data,edng_sigmoid_hashes[logit+16])||!table_hash(s.tanh_lut.data,edng_tanh_hashes[logit+16])||ednx_gru_init(op->handle,&s)) return -1;
        } else if(layout->kind==5) {
            if(aux < -16||aux>8||u64(r+72)!=UINT64_C(0x3e45798ee2308c3a)||u64(r+64)!=edng_epsilon_codes[in+16]) return -1;
            ednx_layer_norm_spec s;s.size=528;s.input_exponent=in;s.gamma_exponent=aux;s.output_exponent=out;
            s.variance_fractional_bits=24;s.epsilon_code=u64(r+64);s.gamma=buffer(data,index,0);s.beta=buffer(data,index,1);
            if(ednx_layer_norm_init(op->handle,&s)) return -1;
        } else {
            op->array=buffer(data,index,0).data;
            if(layout->kind==3&&(aux < -20||aux>4)) return -1;
            if(layout->kind==6&&(out!=-7||!table_hash((const uint8_t *)op->array,edng_tanh_hashes[in+16]))) return -1;
            if(layout->kind==7&&!table_hash((const uint8_t *)op->array,edng_sigmoid_hashes[in+16])) return -1;
            if(layout->kind==8&&!erb_valid((const uint8_t *)op->array)) return -1;
            if(layout->kind==9&&!window_valid((const uint8_t *)op->array)) return -1;
        }
    }
    if(bytes!=align16(cursor)||!zero(data+cursor,bytes-cursor)) return -1;
    /* Grouped hidden concatenations retain the Q7 hidden-state encoding. */
    const unsigned hidden_grids[]={EDNG_GRID_DPGRNN1_INTRA_RNN_OUTPUT,EDNG_GRID_DPGRNN1_INTER_RNN_OUTPUT,
        EDNG_GRID_DPGRNN2_INTRA_RNN_OUTPUT,EDNG_GRID_DPGRNN2_INTER_RNN_OUTPUT};
    for(unsigned i=0;i<4;++i) if(signed_byte(data[192+hidden_grids[i]])!=-7) return -1;
    model->packed=data;model->packed_bytes=bytes;model->grids=(const int8_t *)(data+192);model->magic=EDNG_MAGIC;
    return 0;
}
