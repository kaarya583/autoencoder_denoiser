#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "sdkconfig.h"
#include "audio.h"
#include "frequency_audio.h"

#ifdef CONFIG_EDN_FREQUENCY_MODEL
typedef ednf_model benchmark_model;
typedef ednf_audio_state benchmark_audio_state;
#define MODEL_INIT ednf_init
#define AUDIO_INIT ednf_audio_init
#define AUDIO_PROCESS_PCM16 ednf_audio_process_pcm16
#define NEURAL_REQUIRED_BYTES(model) ((model).workspace_bytes)
#define NEURAL_ALLOCATED_BYTES 49152
#define MODEL_ARCHITECTURE "frequency_unet"
#else
typedef edn_model benchmark_model;
typedef edn_audio_state benchmark_audio_state;
#define MODEL_INIT edn_init
#define AUDIO_INIT edn_audio_init
#define AUDIO_PROCESS_PCM16 edn_audio_process_pcm16
#define NEURAL_REQUIRED_BYTES(model) ((model).state_bytes)
#define NEURAL_ALLOCATED_BYTES 16384
#define MODEL_ARCHITECTURE "global_spectral_tcn"
#endif
#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

extern const uint8_t model_start[] asm("_binary_model_bin_start");
extern const uint8_t model_end[] asm("_binary_model_bin_end");

#define BENCHMARK_FRAMES 1024
#define HOP_MICROSECONDS 16000

static benchmark_model model;
static benchmark_audio_state audio_state; /* Type guarantees 16-byte FFT alignment. */
static uint8_t neural_workspace[NEURAL_ALLOCATED_BYTES];
static int16_t input[256], output[256];
static uint32_t latencies[BENCHMARK_FRAMES];
static volatile int32_t output_checksum;
static const char *TAG = "denoiser_benchmark";

static int compare_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x > y ? 1 : x < y ? -1 : 0;
}

void app_main(void) {
    esp_chip_info_t chip;
    unsigned frame, sample, deadline_misses = 0;
    uint64_t total = 0;
    uint32_t random = 2026, maximum = 0;
    TickType_t wake_time;
    const uint32_t model_memory_caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    const size_t model_bytes = (size_t)(model_end-model_start);
    const uint8_t *model_data = model_start;
    const char *model_storage = "flash_mapped";
    uint8_t *model_copy = NULL;
#ifdef CONFIG_EDN_MODEL_IN_INTERNAL_SRAM
    model_copy = heap_caps_aligned_alloc(16, model_bytes, model_memory_caps);
    if (!model_copy) {
        ESP_LOGE(TAG, "Internal model allocation failed: requested %u bytes, largest block %u",
                 (unsigned)model_bytes,
                 (unsigned)heap_caps_get_largest_free_block(model_memory_caps));
        return;
    }
    memcpy(model_copy, model_start, model_bytes);
    model_data = model_copy;
    model_storage = "internal_sram";
#endif
    if (edn_backend_self_test()) {
        ESP_LOGE(TAG, "Neural backend known-answer self-test failed: %s", edn_neural_backend());
        goto cleanup;
    }
    ESP_LOGI(TAG, "Architecture: %s; neural backend: %s; known-answer self-test passed", MODEL_ARCHITECTURE, edn_neural_backend());
    esp_chip_info(&chip);
    if (MODEL_INIT(&model, model_data, model_bytes) ||
        AUDIO_INIT(&audio_state, &model, neural_workspace, sizeof(neural_workspace))) {
        ESP_LOGE(TAG, "Model or audio initialization failed");
        goto cleanup;
    }
    ESP_LOGI(TAG, "{\"event\":\"start\",\"target\":\"%s\",\"cores\":%d,"
                  "\"configured_cpu_mhz\":%d,\"model_bytes\":%u,\"required_neural_workspace_bytes\":%u,"
                  "\"allocated_neural_workspace_bytes\":%u,"
                  "\"model_storage\":\"%s\",\"runtime_model_copy_bytes\":%u,"
                  "\"free_internal_8bit_heap_after_init\":%u,"
                  "\"largest_internal_8bit_block_after_init\":%u,"
                  "\"audio_state_bytes\":%u,\"sample_rate\":16000,\"hop_samples\":256,"
                  "\"measurement\":\"PCM conversion plus FFT/features/neural/iFFT/OLA; excludes I2S\"}",
             CONFIG_IDF_TARGET, chip.cores, CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ,
             (unsigned)model.model_bytes, (unsigned)NEURAL_REQUIRED_BYTES(model),
             (unsigned)sizeof(neural_workspace), model_storage,
             (unsigned)(model_copy ? model_bytes : 0),
             (unsigned)heap_caps_get_free_size(model_memory_caps),
             (unsigned)heap_caps_get_largest_free_block(model_memory_caps),
             (unsigned)sizeof(audio_state));
    wake_time = xTaskGetTickCount();
    for (frame = 0; frame < BENCHMARK_FRAMES + 32; ++frame) {
        int64_t started, elapsed;
        for (sample = 0; sample < 256; ++sample) {
            float seconds = (float)(frame*256+sample) / 16000.0f;
            random ^= random << 13; random ^= random >> 17; random ^= random << 5;
            input[sample] = (int16_t)(6000*sinf(6.28318530718f*330*seconds) +
                                      2000*((float)(random & 65535)/32768.0f-1));
        }
        started = esp_timer_get_time();
        if (AUDIO_PROCESS_PCM16(&audio_state, input, output)) {
            ESP_LOGE(TAG, "Frame processing failed at %u", frame);
            goto cleanup;
        }
        elapsed = esp_timer_get_time() - started;
        output_checksum = output[frame % 256];
        if (frame >= 32) {
            uint32_t duration = (uint32_t)elapsed;
            latencies[frame-32] = duration;
            total += duration;
            if (duration > maximum) maximum = duration;
            if (duration > HOP_MICROSECONDS) ++deadline_misses;
        }
        /* Run at the intended cadence and let other RTOS tasks execute. */
        vTaskDelayUntil(&wake_time, pdMS_TO_TICKS(16));
    }
    qsort(latencies, BENCHMARK_FRAMES, sizeof(latencies[0]), compare_u32);
    ESP_LOGI(TAG, "{\"event\":\"result\",\"frames\":%d,\"mean_us\":%.3f,"
                  "\"p99_us\":%" PRIu32 ",\"max_us\":%" PRIu32 ",\"deadline_misses\":%u,"
                  "\"rtf\":%.6f,\"minimum_free_internal_heap\":%u,"
                  "\"model_storage\":\"%s\",\"runtime_model_copy_bytes\":%u,"
                  "\"minimum_free_internal_8bit_heap\":%u,"
                  "\"input\":\"synthetic timing stimulus; not an audio quality evaluation\"}",
             BENCHMARK_FRAMES, (double)total/BENCHMARK_FRAMES,
             latencies[(BENCHMARK_FRAMES*99+99)/100-1], maximum, deadline_misses,
             (double)total/BENCHMARK_FRAMES/HOP_MICROSECONDS,
             (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
             model_storage, (unsigned)(model_copy ? model_bytes : 0),
             (unsigned)heap_caps_get_minimum_free_size(model_memory_caps));
cleanup:
    /* Retain the model through all measured frames and release only at exit. */
    heap_caps_free(model_copy);
}
