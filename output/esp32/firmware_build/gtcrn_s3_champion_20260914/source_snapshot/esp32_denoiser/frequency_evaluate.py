"""Waveform adapter for the actual EDNFQ8 integer graph and shared float32 DSP."""
from pathlib import Path
import ctypes
from functools import lru_cache
import hashlib
import shutil
import subprocess
import tempfile

import numpy as np
import torch
from torch import nn

from .frequency_export import IntegerFrequencyDenoiser
from .frequency_model import FrequencyUNet
from .model import SpectralTCN
from .quantization import round_away
from .runtime_cache import runtime_fingerprint


def _frequency_runtime():
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("The frequency C backend requires a C99 compiler")
    source = Path(__file__).resolve().parents[1] / "firmware/esp32_denoiser/frequency.c"
    fingerprint = runtime_fingerprint(source.parent, ('denoiser.c', 'denoiser.h', 'integer_kernels.h', 'frequency.c', 'frequency.h'))
    return _frequency_runtime_for_source(source, compiler, fingerprint)


@lru_cache(maxsize=1)
def _frequency_runtime_for_source(source, compiler, fingerprint):
    # fingerprint is part of the cache key, including transitively included headers.
    directory = tempfile.TemporaryDirectory(prefix="frequency-denoiser-")
    library_path = Path(directory.name) / "frequency.so"
    try:
        subprocess.run([compiler,"-std=c99","-O2","-Wall","-Wextra","-Werror","-shared","-fPIC",
                        str(source),str(source.with_name("denoiser.c")),"-o",str(library_path)],
                       check=True,capture_output=True,text=True)
        library = ctypes.CDLL(str(library_path))
    except Exception:
        directory.cleanup()
        raise
    library.ednf_model_handle_bytes.restype = ctypes.c_size_t
    library.ednf_workspace_bytes.argtypes = [ctypes.c_void_p]
    library.ednf_workspace_bytes.restype = ctypes.c_size_t
    library.ednf_init.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
    library.ednf_reset.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
    library.ednf_process_frame.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,
                                         ctypes.c_void_p,ctypes.c_void_p]
    for name in ("ednf_init","ednf_reset","ednf_process_frame"):
        getattr(library,name).restype = ctypes.c_int
    return library,directory


class CFrequencyNetwork:
    def __init__(self, data):
        self.library,self.directory = _frequency_runtime()
        self.blob = ctypes.create_string_buffer(data)
        self.handle = ctypes.create_string_buffer(self.library.ednf_model_handle_bytes())
        if self.library.ednf_init(self.handle,self.blob,len(data)):
            raise ValueError("The C runtime rejected the frequency model")
        self.workspace_bytes = self.library.ednf_workspace_bytes(self.handle)
        self.workspace = ctypes.create_string_buffer(self.workspace_bytes)
        self.reset()

    def reset(self):
        if self.library.ednf_reset(self.handle,self.workspace,self.workspace_bytes):
            raise RuntimeError("Unable to reset the frequency C workspace")

    def process(self, features):
        if features.dtype != np.int8 or features.ndim != 3 or features.shape[1:] != (3,257):
            raise ValueError("Expected INT8 [frames,3,257] features")
        features = np.ascontiguousarray(features)
        outputs = np.empty((len(features),514),dtype=np.int8)
        for index, frame in enumerate(features):
            if self.library.ednf_process_frame(self.handle,self.workspace,self.workspace_bytes,
                                               frame.ctypes.data,outputs[index].ctypes.data):
                raise RuntimeError(f"Frequency C inference failed at frame {index}")
        return outputs


class FrequencyIntegerWaveformEnhancer(nn.Module):
    """Integer neural inference; there are no floating neural parameters here."""
    frame_features = FrequencyUNet.frame_features
    apply_mask = SpectralTCN.apply_mask

    def __init__(self, source: str | Path | bytes, backend="c"):
        super().__init__()
        if backend not in {"c","numpy"}:
            raise ValueError("backend must be c or numpy")
        metadata = IntegerFrequencyDenoiser(source)
        self.network = CFrequencyNetwork(metadata.data) if backend == "c" else metadata
        self.config = metadata.config
        self.backend = backend
        self.model_bytes = len(metadata.data)
        self.model_sha256 = hashlib.sha256(metadata.data).hexdigest()
        self.input_exponent = metadata.input_exponent
        self.output_exponent = metadata.output_exponent
        self.neural_state_bytes = self.network.workspace_bytes if backend == "c" else None
        self.neural_history_bytes = (2*sum(self.config.local_dilations)*self.config.encoder_channels[-1]*33
                                     + 2*sum(self.config.global_dilations)*self.config.global_width)
        self.register_buffer("window", torch.from_numpy(metadata.window.copy()))

    def forward_features(self, features):
        encoded = round_away(features / (2.0**self.input_exponent)).clamp(-128,127).to(torch.int8)
        outputs = []
        for item in encoded.permute(0,2,1,3).numpy():
            self.network.reset()
            outputs.append(self.network.process(np.ascontiguousarray(item)))
        return torch.from_numpy(np.stack(outputs)).float().transpose(1,2) * (2.0**self.output_exponent)

    @torch.inference_mode()
    def forward(self, noisy):
        if noisy.device.type != "cpu" or not noisy.is_floating_point() or not bool(torch.isfinite(noisy).all()):
            raise ValueError("Integer waveform evaluation requires finite CPU floating-point audio")
        return SpectralTCN.forward(self, noisy.float())
