import copy
import json
import subprocess
import sys

import pytest

from esp32_denoiser.comparison import compare_evaluations


def evaluation(improvements, *, training=False):
    rows = []
    for index, improvement in enumerate(improvements):
        row = {"id": f"item_{index}", "si_sdri": improvement}
        row.update({"noisy_si_sdr": 3.0, "si_sdr": 3 + improvement} if training else
                   {"si_sdr_noisy": 3.0, "si_sdr_enhanced": 3 + improvement})
        rows.append(row)
    return {"per_utterance" if training else "utterances": rows}


def test_mixed_schemas_pair_by_id_and_bootstrap_is_order_independent():
    reference = evaluation([1, 2, 3], training=True)
    candidate = evaluation([2, 1, 6])
    result = compare_evaluations(candidate, reference, bootstrap_samples=1000)
    score = result["metrics"]["si_sdri"]
    assert score["mean_difference"] == 1
    assert score["candidate_mean"] == 3
    assert score["reference_mean"] == 2
    assert score["wins"] == 2 and score["losses"] == 1
    assert score["win_fraction"] == pytest.approx(2 / 3)
    candidate["utterances"].reverse()
    assert compare_evaluations(candidate, reference, bootstrap_samples=1000) == result


@pytest.mark.parametrize("defect", ["duplicate", "ids", "baseline", "invalid_baseline", "improvement", "samples", "manifest_hash"])
def test_unpaired_or_inconsistent_results_fail(defect):
    reference, candidate = evaluation([1, 2]), evaluation([2, 3])
    if defect == "duplicate":
        candidate["utterances"][1]["id"] = "item_0"
    elif defect == "ids":
        candidate["utterances"][1]["id"] = "different"
    elif defect == "baseline":
        candidate["utterances"][0]["si_sdr_noisy"] = 2
        candidate["utterances"][0]["si_sdri"] = 3
    elif defect == "invalid_baseline":
        candidate["utterances"][0]["si_sdr_noisy"] = None
    elif defect == "improvement":
        candidate["utterances"][0]["si_sdri"] = 99
    elif defect == "samples":
        candidate["utterances"][0]["samples"] = 10
        reference["utterances"][0]["samples"] = 11
    else:
        candidate["manifest_sha256"], reference["manifest_sha256"] = "a", "b"
    with pytest.raises(ValueError):
        compare_evaluations(candidate, reference)


def test_common_finite_cohorts_are_metric_specific_and_json_has_no_nan():
    candidate, reference = evaluation([2, 3, 4]), evaluation([1, 2, 3])
    candidate["utterances"][0]["si_sdr_enhanced"] = None
    for document in (candidate, reference):
        for row in document["utterances"]:
            row.update(pesq_wb_noisy=1.0, pesq_wb_enhanced=2.0)
    candidate["utterances"][1]["pesq_wb_enhanced"] = None
    reference["utterances"][2]["pesq_wb_noisy"] = None
    result = compare_evaluations(candidate, reference, bootstrap_samples=1000)
    sdr, pesq = result["metrics"]["si_sdri"], result["metrics"]["pesq_wb_enhanced"]
    assert sdr["common_finite_utterances"] == 2
    assert sdr["confidence_interval_95"] == [1, 1]
    assert pesq["common_finite_utterances"] == 1
    assert pesq["excluded_utterances"] == 2
    assert pesq["confidence_interval_95"] == [0, 0]
    assert pesq["ties"] == 1
    json.dumps(result, allow_nan=False)
    for row in candidate["utterances"]:
        row["si_sdr_enhanced"] = None
    empty = compare_evaluations(candidate, reference)["metrics"]["si_sdri"]
    assert empty["common_finite_utterances"] == 0
    assert empty["mean_difference"] is None and empty["confidence_interval_95"] is None


def test_crop_bootstrap_does_not_treat_repeated_snr_as_independent():
    changes = [-2, -1, 1, 2]
    reference, candidate = evaluation([0] * 20), evaluation([x for x in changes for _ in range(5)])
    manifest = [{"id": f"item_{i}", "suite": "mixtures", "base_crop": i // 5} for i in range(20)]
    grouped = compare_evaluations(candidate, reference, manifest=manifest, cluster_key="base_crop", seed=4)
    base_only = compare_evaluations(evaluation(changes), evaluation([0] * 4), seed=4)
    independent = compare_evaluations(candidate, reference, seed=4)
    grouped_score = grouped["metrics"]["si_sdri"]
    assert grouped_score["bootstrap_units"] == 4
    assert grouped_score["confidence_interval_95"] == base_only["metrics"]["si_sdri"]["confidence_interval_95"]
    assert grouped_score["confidence_interval_95"][1] > independent["metrics"]["si_sdri"]["confidence_interval_95"][1]
    assert "sharing a speaker or recording" in grouped["bootstrap"]["scope"]
    manifest.reverse()
    assert compare_evaluations(candidate, reference, manifest=manifest, cluster_key="base_crop", seed=4) == grouped


def test_manifest_clusters_are_strict_and_suite_names_prevent_collisions():
    reference, candidate = evaluation([0, 0]), evaluation([1, 1])
    manifest = [{"id": "item_0", "suite": "clean", "base_crop": 0},
                {"id": "item_1", "suite": "mixtures", "base_crop": 0}]
    result = compare_evaluations(candidate, reference, manifest=manifest, cluster_key="base_crop")
    assert result["metrics"]["si_sdri"]["bootstrap_units"] == 2
    for malformed in (manifest[:1], [manifest[0], manifest[0]],
                      [manifest[0], {"id": "item_1"}]):
        with pytest.raises(ValueError):
            compare_evaluations(candidate, reference, manifest=malformed, cluster_key="base_crop")
    with pytest.raises(ValueError):
        compare_evaluations(candidate, reference, cluster_key="base_crop")


def test_voicebank_scope_and_cli(tmp_path):
    candidate = evaluation([1, 2])
    for row, speaker in zip(candidate["utterances"], ("p226", "p287")):
        row["id"] = f"{speaker}_001"
    reference = copy.deepcopy(candidate)
    first, second, output = (tmp_path / name for name in ("candidate.json", "reference.json", "result.json"))
    first.write_text(json.dumps(candidate))
    second.write_text(json.dumps(reference))
    process = subprocess.run([sys.executable, "-m", "esp32_denoiser.comparison", "--candidate", str(first),
                              "--reference", str(second), "--output", str(output), "--bootstrap-samples", "1000"],
                             capture_output=True, text=True, check=True)
    result = json.loads(output.read_text())
    assert result == json.loads(process.stdout)
    assert "two held-out speakers p226/p287" in result["bootstrap"]["scope"]
    assert "not an independent-speaker population" in result["bootstrap"]["scope"]


def test_training_exports_that_omit_invalid_ids_cannot_claim_a_complete_pairing():
    training = evaluation([1], training=True)
    training["invalid_utterances"] = 1
    with pytest.raises(ValueError, match="omits invalid IDs"):
        compare_evaluations(training, evaluation([1]))


@pytest.mark.parametrize("role", ("clean", "noisy"))
def test_unchanged_manifest_and_baseline_cannot_hide_changed_audio(role):
    candidate, reference = evaluation([2]), evaluation([1])
    for report in (candidate, reference):
        report["manifest_sha256"] = "a" * 64
        report["utterances"][0]["audio_sha256"] = {"clean": "b" * 64, "noisy": "c" * 64}
    candidate["utterances"][0]["audio_sha256"][role] = "d" * 64
    with pytest.raises(ValueError, match="audio_sha256 differs"):
        compare_evaluations(candidate, reference, bootstrap_samples=100)


def test_historical_reports_remain_comparable_with_explicit_byte_verification_scope():
    candidate, reference = evaluation([2, 3]), evaluation([1, 2])
    hashes = {"clean": "a" * 64, "noisy": "b" * 64}
    for row in candidate["utterances"]:
        row["audio_sha256"] = hashes.copy()
    old = compare_evaluations(candidate, reference, bootstrap_samples=100)
    assert old["verification"]["audio_sha256_matched_utterances"] == 0
    assert old["verification"]["audio_sha256_unverified_utterances"] == 2
    assert old["verification"]["manifest_sha256_matched"] is False
    reference["utterances"][0]["audio_sha256"] = hashes.copy()
    partial = compare_evaluations(candidate, reference, bootstrap_samples=100)
    assert partial["verification"]["audio_sha256_matched_utterances"] == 1
    assert partial["verification"]["audio_sha256_unverified_utterances"] == 1
    reference["utterances"][1]["audio_sha256"] = hashes.copy()
    complete = compare_evaluations(candidate, reference, bootstrap_samples=100)
    assert complete["verification"]["audio_sha256_matched_utterances"] == 2
    assert complete["verification"]["audio_sha256_unverified_utterances"] == 0
    assert complete["metrics"] == partial["metrics"] == old["metrics"]


@pytest.mark.parametrize("malformed", (None, {}, {"clean": "a" * 64}, {"clean": "a" * 64, "noisy": "not-a-hash"}))
def test_present_but_malformed_audio_hashes_are_not_treated_as_historical_omissions(malformed):
    candidate, reference = evaluation([2]), evaluation([1])
    candidate["utterances"][0]["audio_sha256"] = malformed
    with pytest.raises(ValueError, match="Invalid audio_sha256"):
        compare_evaluations(candidate, reference, bootstrap_samples=100)
