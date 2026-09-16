idf_component_register(
    SRCS "graph.c" "packed.c" "audio.c"
         "../experimental_int8/operators.c" "../experimental_int8/primitives.c"
         "../experimental_dsp/gtcrn_erb.c"
    INCLUDE_DIRS "."
    REQUIRES esp-dsp)
target_compile_options(${COMPONENT_LIB} PRIVATE -fstack-usage)
