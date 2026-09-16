#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "sdkconfig.h"
#include "graph.h"
#include "audio.h"
#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "known_answer.h"

extern const uint8_t gtcrn_model_start[], gtcrn_model_end[];
#define BENCHMARK_FRAMES 1024
#define HOP_MICROSECONDS 16000
static int16_t input[256], output[256];
static uint32_t durations[BENCHMARK_FRAMES];
static volatile int32_t checksum;
static const char *TAG = "gtcrn_benchmark";

static uint32_t fnv1a(const int8_t *data,size_t count) {
    uint32_t value=UINT32_C(2166136261);
    for(size_t i=0;i<count;++i) value=(value^(uint8_t)data[i])*UINT32_C(16777619);
    return value;
}
static int known_answer(const edng_model *model,int8_t *state,size_t state_bytes,
                        void *workspace,size_t workspace_bytes) {
    int8_t features[387],mask[258];uint32_t random=2026;
    if(strcmp(GTCRN_MODEL_SHA256,GTCRN_KNOWN_ANSWER_MODEL_SHA256)||edng_reset(model,state,state_bytes)) return -1;
    for(unsigned frame=0;frame<GTCRN_KNOWN_ANSWER_FRAMES;++frame) {
        for(unsigned i=0;i<387;++i) {
            if(frame) {random^=random<<13;random^=random>>17;random^=random<<5;}
            features[i]=frame?(int8_t)((int)(random&255)-128):0;
        }
        if(edng_process_frame(model,state,state_bytes,features,sizeof(features),mask,sizeof(mask),workspace,workspace_bytes)||
           fnv1a(mask,sizeof(mask))!=gtcrn_known_answer[frame][0]||
           fnv1a(state,state_bytes)!=gtcrn_known_answer[frame][1]) return -1;
        vTaskDelay(1);
    }
    return 0;
}

static size_t align16(size_t value) { return (value+15u)&~(size_t)15u; }
static int compare_u32(const void *a,const void *b) {
    uint32_t x=*(const uint32_t *)a,y=*(const uint32_t *)b;
    return x>y?1:x<y?-1:0;
}

void app_main(void) {
    const uint32_t caps=MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT;
    size_t model_bytes=(size_t)(gtcrn_model_end-gtcrn_model_start);
    size_t handle_bytes=edng_handle_bytes(),neural_bytes=edng_state_bytes();
    size_t workspace_bytes=edng_workspace_bytes(),audio_bytes=edng_audio_bytes();
    size_t arena_bytes=align16(handle_bytes)+align16(neural_bytes)+align16(workspace_bytes)+align16(audio_bytes);
    uint8_t *arena=heap_caps_aligned_alloc(16,arena_bytes,caps),*model_copy=NULL;
    const uint8_t *model_data=gtcrn_model_start;
    const char *placement="flash_mapped";
    if(!arena) {
        ESP_LOGE(TAG,"Internal arena allocation failed: requested %u, largest block %u",
                 (unsigned)arena_bytes,(unsigned)heap_caps_get_largest_free_block(caps));
        return;
    }
#ifdef CONFIG_GTCRN_MODEL_IN_INTERNAL_SRAM
    model_copy=heap_caps_aligned_alloc(16,model_bytes,caps);
    if(!model_copy) {ESP_LOGE(TAG,"Internal model copy allocation failed");goto cleanup;}
    memcpy(model_copy,model_data,model_bytes);model_data=model_copy;placement="internal_sram";
#endif
    edng_model *model=(edng_model *)arena;
    int8_t *neural=(int8_t *)(arena+align16(handle_bytes));
    void *workspace=(uint8_t *)neural+align16(neural_bytes);
    edng_audio *audio=(edng_audio *)((uint8_t *)workspace+align16(workspace_bytes));
    if(edng_init(model,handle_bytes,model_data,model_bytes)) {
        ESP_LOGE(TAG,"Packed model initialization failed");goto cleanup;
    }
    if(known_answer(model,neural,neural_bytes,workspace,workspace_bytes)) {
        ESP_LOGE(TAG,"Integer known-answer check failed; refusing timing run");goto cleanup;
    }
    ESP_LOGI(TAG,"{\"event\":\"known_answer_pass\",\"frames\":%d,\"scope\":\"integer mask/state hashes\"}",GTCRN_KNOWN_ANSWER_FRAMES);
    if(edng_audio_init(audio,audio_bytes,model,neural,neural_bytes,workspace,workspace_bytes)) {
        ESP_LOGE(TAG,"Audio initialization failed");goto cleanup;
    }
    esp_chip_info_t chip;esp_chip_info(&chip);
    ESP_LOGI(TAG,"{\"event\":\"start\",\"architecture\":\"gtcrn\",\"neural_backend\":\"portable_scalar_int8\","
                 "\"target\":\"%s\",\"cores\":%d,\"cpu_mhz\":%d,\"packed_bytes\":%u,"
                 "\"packed_sha256\":\"%s\",\"placement\":\"%s\",\"model_copy_bytes\":%u,"
                 "\"model_handle_bytes\":%u,\"neural_state_bytes\":%u,\"neural_workspace_bytes\":%u,"
                 "\"audio_state_bytes\":%u,\"allocated_internal_arena_bytes\":%u,"
                 "\"pcm_buffers_bytes\":%u,\"timing_samples_bytes\":%u,"
                 "\"free_internal_heap_after_init\":%u,\"largest_internal_block_after_init\":%u,"
                 "\"measurement\":\"PCM conversion+FFT+ERB+full INT8 graph+iFFT+OLA; excludes I2S\"}",
             CONFIG_IDF_TARGET,chip.cores,CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ,(unsigned)model_bytes,
             GTCRN_MODEL_SHA256,placement,(unsigned)(model_copy?model_bytes:0),
             (unsigned)handle_bytes,(unsigned)neural_bytes,(unsigned)workspace_bytes,
             (unsigned)audio_bytes,(unsigned)arena_bytes,(unsigned)(sizeof(input)+sizeof(output)),
             (unsigned)sizeof(durations),(unsigned)heap_caps_get_free_size(caps),
             (unsigned)heap_caps_get_largest_free_block(caps));
    uint64_t total=0;uint32_t maximum=0,random=2026;unsigned misses=0;
    TickType_t wake=xTaskGetTickCount();
    for(unsigned frame=0;frame<BENCHMARK_FRAMES+32;++frame) {
        for(unsigned sample=0;sample<256;++sample) {
            float time=(float)(frame*256+sample)/16000.0f;
            random^=random<<13;random^=random>>17;random^=random<<5;
            input[sample]=(int16_t)(6000*sinf(6.28318530718f*330*time)+2000*((float)(random&65535)/32768.0f-1));
        }
        int64_t start=esp_timer_get_time();
        if(edng_audio_process_pcm16(audio,input,output)) {
            ESP_LOGE(TAG,"Audio processing failed at frame %u",frame);goto cleanup;
        }
        uint32_t duration=(uint32_t)(esp_timer_get_time()-start);
        checksum^=output[frame%256];
        if(frame>=32) {
            durations[frame-32]=duration;total+=duration;
            if(duration>maximum) maximum=duration;
            if(duration>HOP_MICROSECONDS) ++misses;
        }
        /* A slow model must still let the idle task run and report its
         * failure to meet deadlines, instead of starving the watchdog. */
        if(duration>=HOP_MICROSECONDS) {vTaskDelay(1);wake=xTaskGetTickCount();}
        else vTaskDelayUntil(&wake,pdMS_TO_TICKS(16));
    }
    qsort(durations,BENCHMARK_FRAMES,sizeof(durations[0]),compare_u32);
    ESP_LOGI(TAG,"{\"event\":\"result\",\"frames\":%d,\"mean_us\":%.3f,\"p99_us\":%" PRIu32 ","
                 "\"max_us\":%" PRIu32 ",\"deadline_misses\":%u,\"rtf\":%.6f,"
                 "\"minimum_free_internal_heap\":%u,\"main_task_stack_high_water_bytes\":%u,"
                 "\"input\":\"synthetic timing stimulus; not an audio quality evaluation\"}",
             BENCHMARK_FRAMES,(double)total/BENCHMARK_FRAMES,durations[(BENCHMARK_FRAMES*99+99)/100-1],
             maximum,misses,(double)total/BENCHMARK_FRAMES/HOP_MICROSECONDS,
             (unsigned)heap_caps_get_minimum_free_size(caps),(unsigned)uxTaskGetStackHighWaterMark(NULL));
cleanup:
    heap_caps_free(model_copy);heap_caps_free(arena);
}
