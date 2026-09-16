#ifndef EDN_INTEGER_KERNELS_H
#define EDN_INTEGER_KERNELS_H
#include <stddef.h>
#include <stdint.h>
/* Internal raw-dot interface. The input must have 32 readable guard bytes after
 * length. SIMD additionally requires 16-byte input alignment; otherwise scalar
 * fallback applies. Weight storage bounds protect pipelined unaligned reads.
 * The caller validates the absolute dot-plus-bias bound fits INT32. */
int32_t edn_dot_product(const int8_t *input, const int8_t *weights, unsigned length,
                        const uint8_t *weight_storage, size_t storage_bytes);
#endif
