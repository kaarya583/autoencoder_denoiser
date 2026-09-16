"""Guard compute-bearing QAT starts and resumes against changed data recipes."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from esp32_denoiser.gtcrn_qat_train import QATTrainConfig, training_dataset, _resume_contract, _check_resume
from test_gtcrn_recurrent_probe import broad_checkpoint


def config(**kwargs):
    return QATTrainConfig('source.pt', 'model.bin', 'audit.json', 'dev.jsonl', 'run', **kwargs)


@pytest.mark.parametrize('kwargs', [dict(epochs=True), dict(batch_size=0), dict(workers=-1),
    dict(max_hours=float('nan')), dict(learning_rate=0), dict(waveform_loss_weight=-1),
    dict(max_steps_per_epoch=0), dict(max_validation_utterances=True), dict(device='mps'),
    dict(min_learning_rate=.01)])
def test_invalid_compute_configuration_fails_before_work(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


def test_training_recipe_uses_parent_probability_and_never_reads_dev_audio(broad_checkpoint):
    from esp32_denoiser.mixtures import HybridTrainingDataset
    _, parent, development = broad_checkpoint
    parent=deepcopy(parent)
    parent['train_config']['clean_identity_probability']=.15
    parent['provenance']['clean_identity_probability']=.15
    for row in map(json.loads, development.read_text().splitlines()):
        for role in ('clean', 'noisy'):
            Path(row[role]).write_bytes(b'development audio must not be loaded')
    data, recipe=training_dataset(parent, development)
    assert isinstance(data, HybridTrainingDataset)
    assert recipe['clean_identity_probability'] == .15
    assert data.paired.clean_identity_prob == .15
    assert data.synthetic.clean_identity_prob == .15
    assert recipe['samples_per_epoch'] == len(data)
    assert recipe['manifests']['development']['sha256']


def test_changed_training_manifest_and_final_test_rejected(broad_checkpoint):
    _, parent, development = broad_checkpoint
    train=Path(parent['train_config']['train_manifest']); original=train.read_bytes()
    train.write_bytes(original+b'\n')
    with pytest.raises(ValueError, match='train manifest hash'):
        training_dataset(parent, development)
    train.write_bytes(original)
    rows=[dict(json.loads(line),source_split='test') for line in development.read_text().splitlines()]
    final_manifest=development.with_name('sealed_final.jsonl')
    final_manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError, match='test remains sealed'):
        training_dataset(parent, final_manifest)


def test_resume_allows_budget_extension_but_rejects_recipe_and_optimizer_changes():
    initial=config(); provenance=dict(source_sha256='frozen', recipe={'clean_identity_probability':.03})
    contract=_resume_contract(initial, provenance)
    saved=dict(phase='qat', resume_contract=contract, optimizer={}, scheduler={}, epoch=3,
               best_score=8, stale=0, best_candidate={})
    _check_resume(saved, _resume_contract(replace(initial, epochs=60,max_hours=12,resume='last.pt',workers=0),provenance))
    for changed in (replace(initial,batch_size=8), replace(initial,seed=17), replace(initial,learning_rate=2e-4),
                    replace(initial,max_validation_utterances=4)):
        with pytest.raises(ValueError, match='resume configuration'):
            _check_resume(saved, _resume_contract(changed,provenance))
    with pytest.raises(ValueError, match='resume configuration'):
        _check_resume(saved, _resume_contract(initial,dict(provenance,source_sha256='different')))
    del saved['optimizer']
    with pytest.raises(ValueError, match='complete optimizer checkpoint'):
        _check_resume(saved, contract)
