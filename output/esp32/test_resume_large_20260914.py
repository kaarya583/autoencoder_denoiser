"""Small control-flow checks; no training or GPU work is performed."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).with_name("resume_large_20260914.py")
SPEC = importlib.util.spec_from_file_location("resume_large", PATH)
coordinator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coordinator)


def test_resume_preserves_all_scientific_settings_and_requests_optimizer_restore(tmp_path):
    saved = {"epoch": 158, "train_config": {"phase": "float", "learning_rate": .0002,
             "min_learning_rate": .00001, "patience": 60, "seed": 2026, "epoch_samples": 43208,
             "batch_size": 48, "model_options": {"normalize_input": True}, "clean_identity_probability": .03,
             "resume_optimizer": False, "resume": "parent.pt", "output_dir": "original", "epochs": 240, "max_hours": 12}}
    before = copy.deepcopy(saved)
    result = coordinator.resume_config(saved, tmp_path, 4.75)
    assert saved == before
    changed = {key for key in result if result[key] != saved["train_config"][key]}
    assert changed == {"resume", "resume_optimizer", "output_dir", "max_hours"}
    assert result["resume"] == str(tmp_path / "last.pt")
    assert result["resume_optimizer"] is True
    assert result["epochs"] == 240


@pytest.mark.parametrize("watcher_code", [0, 1])
def test_successful_watcher_exit_before_trainer_exit_is_allowed(watcher_code, monkeypatch):
    supervisor = coordinator.Supervisor(SimpleNamespace())
    training, watcher = object(), object()
    statuses = iter([None, None, 0, 0])
    monkeypatch.setattr(supervisor, "poll", lambda job: watcher_code if job is watcher else next(statuses))
    monkeypatch.setattr(coordinator.time, "sleep", lambda *_: None)
    if watcher_code:
        with pytest.raises(RuntimeError, match="watcher failed"):
            supervisor.wait(training, coordinator.time.monotonic() + 10, companion=watcher)
    else:
        supervisor.wait(training, coordinator.time.monotonic() + 10, companion=watcher)


def test_pid_reuse_never_signals_unrelated_job(monkeypatch):
    old = {"pid": 123, "pgid": 123, "session_id": 123, "start_ticks": 50}
    monkeypatch.setattr(coordinator, "process_identity", lambda _: {**old, "start_ticks": 51})
    monkeypatch.setattr(coordinator.os, "killpg", lambda *_: pytest.fail("Unrelated process was signalled"))
    coordinator.stop_owned(old)


def test_artifacts_skip_only_with_matching_inputs_and_bytes(tmp_path):
    artifact = tmp_path / "model.bin"
    marker = tmp_path / "calibration.complete.json"
    artifact.write_bytes(b"partial")
    assert not coordinator.finished_stage(marker, {"checkpoint": "one"})
    coordinator.mark_stage(marker, {"checkpoint": "one"}, [artifact])
    assert coordinator.finished_stage(marker, {"checkpoint": "one"})
    with pytest.raises(ValueError, match="input identity"):
        coordinator.finished_stage(marker, {"checkpoint": "two"})
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="modified"):
        coordinator.finished_stage(marker, {"checkpoint": "one"})


def test_previous_configs_and_snapshots_cannot_be_overwritten(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "backup"
    source.write_bytes(b"original")
    coordinator.copy_immutable(source, destination)
    coordinator.copy_immutable(source, destination)
    source.write_bytes(b"new")
    with pytest.raises(ValueError, match="Immutable output differs"):
        coordinator.copy_immutable(source, destination)
    assert destination.read_bytes() == b"original"


def test_complete_evaluation_checks_actual_ids_and_metric_denominators(tmp_path):
    manifest, report_path = tmp_path / "manifest.jsonl", tmp_path / "report.json"
    manifest.write_text('{"id":"a"}\n{"id":"b"}\n')
    report = {"manifest_sha256": coordinator.sha(manifest), "limited_evaluation": False,
              "utterances": [{"id": "a"}, {"id": "b"}],
              "summary": {"utterances": 2, "valid_utterances": 2, "invalid_utterances": 0, "si_sdri": 5.5},
              "perceptual": {"summary": {"pesq_wb": {"valid_utterances": 1, "invalid_utterances": 1},
                                          "stoi": {"valid_utterances": 2, "invalid_utterances": 0}}}}
    coordinator.write_json(report_path, report)
    assert coordinator.validate_evaluation(report_path, manifest, 2)["si_sdri"] == 5.5
    report["utterances"][1]["id"] = "a"
    coordinator.write_json(report_path, report)
    with pytest.raises(ValueError, match="complete finite"):
        coordinator.validate_evaluation(report_path, manifest, 2)


def test_partial_history_is_not_silently_appended(tmp_path):
    history = tmp_path / "history.jsonl"
    history.write_text('{"epoch":157}\n{"epoch":158')
    with pytest.raises(ValueError, match="Incomplete JSONL"):
        coordinator.records(history)


def test_global_lock_prevents_duplicate_supervisors(tmp_path):
    with coordinator.exclusive(tmp_path / "lock"):
        with pytest.raises(RuntimeError, match="already active"):
            with coordinator.exclusive(tmp_path / "lock"):
                pytest.fail("Duplicate supervisor acquired lock")


def test_pinned_source_inventory_matches_reviewed_bundle():
    root = PATH.parents[2]
    bundle = root / "output/esp32/source_bundles" / (coordinator.BUNDLE_SHA + ".zip")
    result = coordinator.verify_source(root, bundle)
    assert result["verified_installed_files"] == 96
