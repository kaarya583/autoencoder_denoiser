"""Complete frequency-model C audio using the shared streaming/PCM16 adapter."""
import ctypes
from functools import lru_cache
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile

from torch import nn

from .embedded import EmbeddedWaveformEnhancer
from .frequency_export import IntegerFrequencyDenoiser
from .runtime_cache import runtime_fingerprint


_WRAPPER = r"""
#define _POSIX_C_SOURCE 200112L
#include "frequency_audio.h"
#include <stdlib.h>
typedef struct { ednf_model model; ednf_audio_state audio; void *neural; } host_context;
void *edn_host_create(const void *blob, size_t bytes) {
    host_context *context;
    if (posix_memalign((void **)&context, 16, sizeof(*context))) return NULL;
    context->neural = NULL;
    if (ednf_init(&context->model, blob, bytes)) { free(context); return NULL; }
    size_t needed = ednf_workspace_bytes(&context->model);
    context->neural = calloc(1, needed);
    if (!context->neural || ednf_audio_init(&context->audio, &context->model, context->neural, needed)) {
        free(context->neural); free(context); return NULL;
    }
    return context;
}
void edn_host_destroy(void *handle) {
    host_context *context = handle;
    if (context) { free(context->neural); free(context); }
}
size_t edn_host_neural_bytes(void *handle) {
    return ednf_workspace_bytes(&((host_context *)handle)->model);
}
int edn_host_reset(void *handle) { return ednf_audio_reset(&((host_context *)handle)->audio); }
int edn_host_float(void *handle, const float *input, float *output) {
    return ednf_audio_process(&((host_context *)handle)->audio, input, output);
}
int edn_host_pcm16(void *handle, const int16_t *input, int16_t *output) {
    return ednf_audio_process_pcm16(&((host_context *)handle)->audio, input, output);
}
"""


def _compiled_frequency_audio():
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("Complete C frequency audio requires a C99 compiler")
    source = Path(__file__).resolve().parents[1] / "firmware/esp32_denoiser"
    fingerprint = runtime_fingerprint(source, ('denoiser.c', 'denoiser.h', 'integer_kernels.h', 'frequency.c', 'frequency.h', 'frequency_audio.c', 'frequency_audio.h', 'audio_dsp.h'), _WRAPPER)
    return _compiled_frequency_audio_for_source(source, compiler, fingerprint)


@lru_cache(maxsize=1)
def _compiled_frequency_audio_for_source(source, compiler, fingerprint):
    # fingerprint is part of the cache key, including transitively included headers.
    directory = tempfile.TemporaryDirectory(prefix="frequency-audio-")
    wrapper = Path(directory.name) / "host.c"
    library_path = Path(directory.name) / "audio.so"
    wrapper.write_text(_WRAPPER)
    try:
        subprocess.run([compiler,"-std=c99","-O2","-Wall","-Wextra","-Werror","-shared","-fPIC",
                        "-I",str(source),str(wrapper),str(source/"denoiser.c"),str(source/"frequency.c"),
                        str(source/"frequency_audio.c"),"-lm","-o",str(library_path)],
                       check=True,capture_output=True,text=True)
        library = ctypes.CDLL(str(library_path))
    except Exception:
        directory.cleanup()
        raise
    library.edn_host_create.argtypes = [ctypes.c_void_p,ctypes.c_size_t]
    library.edn_host_create.restype = ctypes.c_void_p
    library.edn_host_destroy.argtypes = [ctypes.c_void_p]
    library.edn_host_destroy.restype = None
    library.edn_host_neural_bytes.argtypes = [ctypes.c_void_p]
    library.edn_host_neural_bytes.restype = ctypes.c_size_t
    library.ednf_audio_state_bytes.restype = ctypes.c_size_t
    library.edn_host_reset.argtypes = [ctypes.c_void_p]
    library.edn_host_reset.restype = ctypes.c_int
    for name in ("edn_host_float","edn_host_pcm16"):
        getattr(library,name).argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
        getattr(library,name).restype = ctypes.c_int
    return library,directory


class FrequencyEmbeddedWaveformEnhancer(EmbeddedWaveformEnhancer):
    def __init__(self, source, io_format="pcm16"):
        nn.Module.__init__(self)
        if io_format not in {"float32","pcm16"}:
            raise ValueError("io_format must be float32 or pcm16")
        metadata = IntegerFrequencyDenoiser(source)
        self.io_format = io_format
        self.sample_rate = metadata.config.sample_rate
        self.hop_length = metadata.config.hop_length
        self.model_bytes = len(metadata.data)
        self.model_sha256 = hashlib.sha256(metadata.data).hexdigest()
        self.library,self.directory = _compiled_frequency_audio()
        self.blob = ctypes.create_string_buffer(metadata.data)
        self.handle = self.library.edn_host_create(self.blob,len(metadata.data))
        if not self.handle:
            raise ValueError("The complete C frequency frontend rejected the model")
        self.neural_state_bytes = self.library.edn_host_neural_bytes(self.handle)
        self.host_audio_state_bytes = self.library.ednf_audio_state_bytes()
        self.reset_statistics()
