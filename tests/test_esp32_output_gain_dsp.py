"""The serialized level correction must precede PCM clipping and preserve the neural graph."""
import shutil
import struct

import numpy as np
import pytest
import torch

from esp32_denoiser.embedded import EmbeddedWaveformEnhancer
from esp32_denoiser.evaluate import IntegerWaveformEnhancer, _CIntegerNetwork
from esp32_denoiser.export import HEADER, IntegerDenoiser, export_model, with_output_gain
from esp32_denoiser.model import SpectralTCN, SpectralTCNConfig
from esp32_denoiser.quantization import configure_qat


def _source(tmp_path):
    torch.manual_seed(812)
    model = SpectralTCN(SpectralTCNConfig(width=8, dilations=(1, 2)))
    with torch.no_grad():
        model.head.bias[:257].fill_(0.5)  # Deliberate 2x gain exposes post-clip mistakes.
    model = configure_qat(model).eval()
    path = tmp_path / 'original.bin'
    export_model(model, path)
    return model, path


def test_gain_serialization_preserves_neural_payload_and_restores_legacy_bytes(tmp_path):
    model, source = _source(tmp_path)
    target = tmp_path / 'corrected.bin'
    report = with_output_gain(source, target, .7654321234)
    before, after = IntegerDenoiser(source), IntegerDenoiser(target)
    assert after.output_gain == float(np.float32(.7654321234))
    assert report['model_bytes'] == len(before.data) + 4
    assert before.output_gain == 1 and before.dsp_version == 1 and after.dsp_version == 2
    assert before.data[HEADER.size:before.dsp_offset] == after.data[HEADER.size:after.dsp_offset]
    assert before.state_bytes == after.state_bytes
    direct = tmp_path / 'direct.bin'
    export_model(model, direct, output_gain=.7654321234)
    assert direct.read_bytes() == target.read_bytes()
    restored = tmp_path / 'restored.bin'
    with_output_gain(target, restored, 1.0)
    assert restored.read_bytes() == source.read_bytes()
    with pytest.raises(ValueError, match='exceeding'):
        with_output_gain(source, tmp_path / 'oversize.bin', .5, max_bytes=len(before.data))
    assert not (tmp_path / 'oversize.bin').exists()


@pytest.mark.parametrize('gain', [0, -1, float('nan'), float('inf'), 8.001, 1e-100, True, [1], '1'])
def test_gain_rejects_invalid_or_unrepresentable_values(tmp_path, gain):
    model, source = _source(tmp_path)
    with pytest.raises(ValueError):
        with_output_gain(source, tmp_path / 'bad.bin', gain)
    with pytest.raises(ValueError):
        export_model(model, tmp_path / 'bad.bin', output_gain=gain)
    assert not (tmp_path / 'bad.bin').exists()


@pytest.mark.skipif(shutil.which('cc') is None, reason='C99 compiler required')
@pytest.mark.parametrize('bits', [0, 0x80000000, 0xbf800000, 0x7f800000, 0x7fc00000, 0x41000001])
def test_numpy_and_c_reject_corrupted_gain_metadata(tmp_path, bits):
    _, source = _source(tmp_path)
    target = tmp_path / 'corrected.bin'
    with_output_gain(source, target, .5)
    blob = bytearray(target.read_bytes())
    struct.pack_into('<I', blob, len(blob)-4, bits)
    with pytest.raises(ValueError):
        IntegerDenoiser(bytes(blob))
    with pytest.raises(ValueError, match='rejected'):
        _CIntegerNetwork(bytes(blob))


@pytest.mark.skipif(shutil.which('cc') is None, reason='C99 compiler required')
def test_full_c_gain_is_after_ola_before_pcm_clipping_with_reset_and_flush(tmp_path):
    _, source = _source(tmp_path)
    target = tmp_path / 'corrected.bin'
    with_output_gain(source, target, .5)
    original = EmbeddedWaveformEnhancer(source, 'float32')
    corrected = EmbeddedWaveformEnhancer(target, 'float32')
    pcm = EmbeddedWaveformEnhancer(target, 'pcm16')
    reference = IntegerWaveformEnhancer(target)
    try:
        t = torch.arange(1537)
        audio = (0.76 * torch.sin(t * .071)).unsqueeze(0)
        audio[0, 0] = 1.25
        encoded = (audio * 32768).round().clamp(-32768, 32767) / 32768
        full_float = corrected(encoded)
        torch.testing.assert_close(full_float, original(encoded) * .5, atol=0, rtol=0)
        torch.testing.assert_close(full_float, reference(encoded), atol=2e-6, rtol=2e-5)
        expected = (full_float * 32768).round().clamp(-32768, 32767) / 32768
        torch.testing.assert_close(pcm(audio), expected, atol=0, rtol=0)
        torch.testing.assert_close(pcm(audio), expected, atol=0, rtol=0)
        assert pcm.io_statistics['input_clipped_samples'] == 2
        # Scaling a previously clipped output loses the peaks; this is deliberately different.
        wrong = (original(encoded) * 32768).round().clamp(-32768, 32767) / 32768 * .5
        assert (wrong - expected).abs().max() > .2
        torch.testing.assert_close(corrected(torch.zeros(1, 257)), torch.zeros(1, 257), atol=0, rtol=0)
    finally:
        original.close()
        corrected.close()
        pcm.close()


@pytest.mark.skipif(shutil.which('cc') is None, reason='C99 compiler required')
def test_gain_preserves_actual_c_neural_frame_outputs(tmp_path):
    _, source = _source(tmp_path)
    target = tmp_path / 'corrected.bin'
    with_output_gain(source, target, 8)
    inputs = np.random.default_rng(916).integers(-128, 128, (57, 387), dtype=np.int8)
    first, second = _CIntegerNetwork(source.read_bytes()), _CIntegerNetwork(target.read_bytes())
    np.testing.assert_array_equal(first.process(inputs), second.process(inputs))


@pytest.mark.skipif(shutil.which('cc') is None, reason='C99 compiler required')
@pytest.mark.parametrize('mutation', ['unknown_version', 'magic_mismatch', 'missing_gain', 'trailing_bytes'])
def test_numpy_and_c_enforce_versioned_dsp_extent(tmp_path, mutation):
    _, source = _source(tmp_path)
    target = tmp_path / 'corrected.bin'
    with_output_gain(source, target, .5)
    metadata = IntegerDenoiser(target)
    blob = bytearray(metadata.data)
    if mutation == 'unknown_version':
        blob[23] = 3
    elif mutation == 'magic_mismatch':
        blob[metadata.dsp_offset:metadata.dsp_offset+4] = b'DSP1'
    elif mutation == 'missing_gain':
        del blob[-4:]
        struct.pack_into('<I', blob, 24, len(blob))
    else:
        blob.extend(b'\0\0\0\0')
        struct.pack_into('<I', blob, 24, len(blob))
    with pytest.raises(ValueError):
        IntegerDenoiser(bytes(blob))
    with pytest.raises(ValueError, match='rejected'):
        _CIntegerNetwork(bytes(blob))
