from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import pytest
import torch
from torch import nn

from esp32_denoiser.gtcrn_equalization import (
    EqualizationConfig, _divide_input_channels, equalize_gtcrn,
    prepare_equalized_checkpoint,
)
from esp32_denoiser.gtcrn_model import GTCRNConfig, GTCRNDenoiser


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


@pytest.mark.parametrize("kind", [nn.Conv2d, nn.ConvTranspose2d])
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_native_grouped_input_axes_preserve_convolution_and_bias(kind, groups):
    torch.manual_seed(92)
    layer = kind(4, 8, (3, 3), padding=(1, 1), groups=groups).eval()
    changed = deepcopy(layer)
    scales = torch.tensor([.125, 2., 8., .5])
    _divide_input_channels(changed, scales)
    x = torch.randn(2, 4, 5, 7)
    torch.testing.assert_close(changed(x*scales[None, :, None, None]), layer(x), rtol=0, atol=0)
    assert torch.equal(layer.bias, changed.bias)


def source():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(93)
        model = GTCRNDenoiser(GTCRNConfig(normalize_input=True)).eval()
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.running_mean.uniform_(-.05, .05)
                    layer.running_var.uniform_(.7, 1.3)
                    layer.weight.uniform_(-1.3, 1.3)
                    layer.bias.uniform_(-.1, .1)
    return model


def test_model_function_streaming_rng_and_source_immutability():
    model = source()
    snapshot = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.random.get_rng_state()
    changed, report = equalize_gtcrn(model)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in snapshot.items())
    assert report['changed_tensors'] and report['source_state_sha256'] != report['transformed_state_sha256']
    assert len(report['cumulative_exponents']) == 12
    assert report['waveform_parity']['streaming_max_abs_error'] < 3e-6
    assert report['data_fitting'] is False
    for name, exponents in report['cumulative_exponents'].items():
        assert len(exponents) == 16 and max(abs(value) for value in exponents) <= 4
    for name, value in snapshot.items():
        if any(term in name for term in ('tra.', 'dpgrnn', 'running_', '.point_bn2.', 'erb.', 'window')):
            assert torch.equal(value, changed.state_dict()[name])


def test_checkpoint_is_new_initialization_preserving_training_recipe(tmp_path):
    model = source()
    original, output = tmp_path/'source.pt', tmp_path/'equalized.pt'
    provenance = {'test_used_for_selection': False, 'source_sha256': {'data.py': 'f'*64},
                  'manifest_sha256': {'train': 'a'*64}, 'resume_checkpoint_sha256': None}
    recipe = {'train_manifest': 'never-open-train.jsonl', 'val_manifest': 'never-open-development.jsonl'}
    torch.save(dict(model=model.state_dict(), model_kind='gtcrn', model_config=asdict(model.config),
                    phase='float', epoch=15, provenance=provenance, train_config=recipe,
                    optimizer={'stale': True}, calibration={'stale': True}), original)
    content = original.read_bytes(); digest = hashlib.sha256(content).hexdigest()
    report = prepare_equalized_checkpoint(original, output, expected_source_sha256=digest)
    saved = torch.load(output, weights_only=False)
    assert original.read_bytes() == content
    assert saved['initialization_only'] and saved['epoch'] == 0
    assert saved['train_config'] == recipe
    assert saved['provenance']['source_sha256'] == provenance['source_sha256']
    assert saved['provenance']['resume_checkpoint_sha256'] == digest
    assert saved['provenance']['channel_equalization']['source_checkpoint_epoch'] == 15
    assert not {'optimizer', 'scheduler', 'scaler', 'calibration'} & saved.keys()
    assert json.loads(output.with_suffix('.equalization.json').read_text()) == report
    reloaded = GTCRNDenoiser(model.config).eval(); reloaded.load_state_dict(saved['model'], strict=True)
    x = torch.randn(1, 769)*.03
    torch.testing.assert_close(reloaded(x), model(x), rtol=3e-5, atol=3e-6)
    with pytest.raises(FileExistsError):
        prepare_equalized_checkpoint(original, output)
    with pytest.raises(ValueError, match='SHA256'):
        prepare_equalized_checkpoint(original, tmp_path/'wrong.pt', expected_source_sha256='0'*64)
    with pytest.raises(ValueError, match='stack equalization'):
        prepare_equalized_checkpoint(output, tmp_path/'stacked.pt')
    assert not (tmp_path/'wrong.pt').exists() and not (tmp_path/'stacked.pt').exists()


def test_invalid_configs_and_training_mode_rejected():
    for values in ({'passes': True}, {'passes': 0}, {'max_channel_exponent': 99}):
        with pytest.raises(ValueError): EqualizationConfig(**values)
    with pytest.raises(ValueError, match='eval'):
        equalize_gtcrn(source().train())
