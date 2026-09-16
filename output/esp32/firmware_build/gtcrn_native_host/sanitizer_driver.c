
#include "graph.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
int main(int argc,char**argv){
 if(argc!=2)return 2;
 FILE*f=fopen(argv[1],"rb");if(!f)return 3;
 uint8_t*blob=calloc(99001,1);size_t n=fread(blob,1,99001,f);fclose(f);
 void*model=calloc(1,edng_handle_bytes());
 size_t sn=edng_state_bytes(),wn=edng_workspace_bytes();
 int8_t*st=malloc(sn+32),*ws=malloc(wn+32),out[258],first[258],in[387];
 if(!blob||!model||!st||!ws)return 4;
 memset(st,85,sn+32);memset(ws,85,wn+32);
 if(edng_init(model,edng_handle_bytes(),blob,n)||edng_reset(model,st+16,sn))return 5;
 for(unsigned j=0;j<387;++j)in[j]=(int8_t)((j*37u)%256u-128);
 for(unsigned frame=0;frame<96;++frame){
  if(edng_process_frame(model,st+16,sn,in,387,out,258,ws+16,wn))return 6;
  if(!frame)memcpy(first,out,258);
  for(unsigned j=0;j<16;++j)if(st[j]!=85||st[sn+16+j]!=85||ws[j]!=85||ws[wn+16+j]!=85)return 7;
 }
 if(edng_reset(model,st+16,sn)||edng_process_frame(model,st+16,sn,in,387,out,258,ws+16,wn)||memcmp(out,first,258))return 8;
 for(size_t j=0;j<n;j+=199){blob[j]^=1;if(!edng_init(model,edng_handle_bytes(),blob,n))return 9;blob[j]^=1;}
 free(ws);free(st);free(model);free(blob);puts("96 frames, guard bytes, reset and corruption rejection passed");return 0;
}
