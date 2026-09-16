set(edn_sources "denoiser.c" "audio.c" "frequency.c" "frequency_audio.c")
if(CONFIG_IDF_TARGET_ESP32S3 AND CONFIG_EDN_S3_SIMD_DOT)
    list(APPEND edn_sources "esp_nn_dot_s8_esp32s3.S")
endif()
idf_component_register(SRCS ${edn_sources}
                       INCLUDE_DIRS "."
                       REQUIRES esp-dsp)
