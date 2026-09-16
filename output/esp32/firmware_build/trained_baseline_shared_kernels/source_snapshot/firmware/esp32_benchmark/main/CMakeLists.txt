if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/model.bin")
    message(FATAL_ERROR "Copy an exported EDNSI8 model to main/model.bin before building")
endif()
idf_component_register(SRCS "main.c" INCLUDE_DIRS "."
                       REQUIRES esp32_denoiser esp_timer esp_hw_support
                       EMBED_FILES "model.bin")
