"""Queue one six-hour fixed-grid QAT run from the better NEW large INT8 export.

No training implementation changes or final-test reads. Run in Colab after the
large-branch coordinator, or let this script wait for both complete exports.
Existing experiments and the historical champion are never overwritten.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

BRANCHES = ("float_gtcrn_broad_large", "float_gtcrn_broad_normalized_large")
HISTORICAL_SHA = "dd9c1c83a44bea891fad7d47591578b57a4b3d0d54f2942a17e5402d4ca93e3f"
HISTORICAL_SCORE = 9.202213762239209
PINNED_RELEASE_SHA = "eceb96444d0988d487f7a087190b5baf5437d54f949190dbf0ef67ec1e6a64d3"
PINNED_CODE_SHA = "c73bfebfdf9e57cc0b5e644bdb7e18319cbed4c9f28a3a1a4855afcc5596ab99"
COHORTS = {
    "external": ("/content/extra_audio/development/mixtures.jsonl", 500,
                 "b82497d1febcfba26e1449ac57596e8f6983071e82d48482a1d6de00a5a84742"),
    "clean": ("/content/extra_audio/development/clean.jsonl", 100,
              "5095c806d052665f14ca8c2ca8789caaf6020c425e80f8daae752fcd4db6c808"),
    "primary": ("/content/voicebank/manifests/val.jsonl", 770,
                "c1e0ccf95766f1542da3e8eebc1cf33107ca6ecf3ccf59c7a2d8078c171f5668"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def code_inventory(project):
    # This optional standalone ablation was added locally after the release;
    # it is neither imported nor executed by this QAT/evaluation pipeline.
    paths = [path for path in project.glob("esp32_denoiser/**/*.py") if path.name != "gtcrn_equalization.py"]
    for directory in ("experimental_gtcrn", "experimental_int8", "experimental_dsp", "esp32_denoiser"):
        paths.extend(path for path in (project/"firmware"/directory).rglob("*") if path.suffix in {".c", ".h"})
    inventory = {str(path.relative_to(project)): sha(path) for path in sorted(paths)}
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != PINNED_CODE_SHA:
        raise ValueError("Scientific source differs from the pinned QAT/evaluation implementation")
    return inventory


def verified_report(path, model_sha, cohort, *, perceptual=True):
    report = read(path)
    _, count, manifest_sha = COHORTS[cohort]
    summary, stats = report["summary"], report["model_stats"]
    if (report.get("manifest_sha256") != manifest_sha or report.get("limited_evaluation") is not False
            or stats.get("packed_sha256") != model_sha or stats.get("io_format") != "pcm16"
            or not 0 < stats.get("packed_bytes", 0) <= 99000
            or summary.get("utterances") != count or summary.get("valid_utterances") != count
            or summary.get("invalid_utterances") != 0
            or summary.get("weighting") != "equal per utterance"
            or summary.get("metric") != "zero-mean SI-SDR, capped +/-80 dB"):
        raise ValueError(f"Unmatched model/manifest/precision/valid denominator: {path}")
    rows = report["utterances"]
    if len(rows) != count or len({row["id"] for row in rows}) != count:
        raise ValueError("Evaluation has missing or duplicate utterances")
    score = summary["si_sdri"]
    if not math.isfinite(score) or not math.isclose(score, math.fsum(row["si_sdri"] for row in rows)/count, abs_tol=1e-10, rel_tol=0):
        raise ValueError("Evaluation score is not the reported equal-utterance mean")
    if perceptual and not all(name in report.get("perceptual", {}).get("summary", {}) for name in ("pesq_wb", "stoi")):
        raise ValueError("Full perceptual evaluation has not finished")
    return report


def identities(report):
    return [(row["id"], row["samples"], row["audio_sha256"]["clean"],
             row["audio_sha256"]["noisy"], row["si_sdr_noisy"]) for row in report["utterances"]]


def same_inputs(reference, other):
    if identities(reference) != identities(other):
        raise ValueError("Compared evaluations differ in audio bytes, ordering, length or noisy baseline")


def choose_seed(branches):
    if {branch["name"] for branch in branches} != set(BRANCHES) or len(branches) != 2:
        raise ValueError("QAT seed must be one of exactly the two completed large branches")
    same_inputs(branches[0]["reports"]["external"], branches[1]["reports"]["external"])
    return max(branches, key=lambda branch: branch["reports"]["external"]["summary"]["si_sdri"])


def compare_champion(historical, candidate):
    same_inputs(historical, candidate)
    difference = candidate["summary"]["si_sdri"]-historical["summary"]["si_sdri"]
    return dict(outcome="win" if difference > 0 else "tie" if difference == 0 else "loss",
                improvement_db=difference, replace_historical_champion=difference > 0)


def global_champion(candidates):
    """Historical, new large export, then QAT; a tie keeps the incumbent."""
    if [candidate["kind"] for candidate in candidates] != ["historical", "large_export", "qat"]:
        raise ValueError("Champion comparison requires historical, large export and QAT in that order")
    champion = candidates[0]
    for candidate in candidates[1:]:
        same_inputs(champion["report"], candidate["report"])
        if candidate["report"]["summary"]["si_sdri"] > champion["report"]["summary"]["si_sdri"]:
            champion = candidate
    return {**{key: value for key, value in champion.items() if key != "report"},
            "si_sdri": champion["report"]["summary"]["si_sdri"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/content/esp32_project"))
    parser.add_argument("--runs", type=Path, default=Path("/content/esp32_runs"))
    parser.add_argument("--wait-hours", type=float, default=24)
    args = parser.parse_args()
    if not math.isfinite(args.wait_hours) or args.wait_hours < 0:
        parser.error("wait-hours must be finite and nonnegative")
    project, runs = args.project.resolve(), args.runs.resolve()
    sys.path.insert(0, str(project))
    source_inventory = code_inventory(project)
    from esp32_denoiser.gtcrn_qat_train import QATTrainConfig, training_dataset
    from esp32_denoiser.gtcrn_integer_export import load_gtcrn_integer
    from esp32_denoiser.gtcrn_recurrent_probe import _development_records
    import torch

    # Bind actual dataset files before waiting or spending GPU time.
    manifests = {}
    for cohort, (manifest, count, digest) in COHORTS.items():
        if sha(manifest) != digest or len(_development_records(manifest)) != count:
            raise ValueError(f"Unexpected {cohort} development manifest")
        manifests[manifest] = digest
    completion = runs/"resume_large_20260914/complete.json"
    deadline = time.monotonic()+args.wait_hours*3600
    while not completion.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("Both resumed large branches must finish export and all three evaluations")
        time.sleep(min(60, max(.01, deadline-time.monotonic())))
    completed = read(completion)
    finished = completed.get("branches", [])
    if (completed.get("pipeline_complete") is not True or len(finished) != 2
            or {branch.get("run") for branch in finished} != set(BRANCHES)
            or not all(branch.get("pipeline_complete") is True for branch in finished)):
        raise ValueError("Resume coordinator has not completed both new deployment pipelines")
    frozen_hashes = {str(completion): sha(completion)}
    branches = []
    for name in BRANCHES:
        branch = next(branch for branch in finished if branch["run"] == name)
        deploy = Path(branch["deployment"]).resolve()
        if deploy != (runs/name/"deployment_resume_20260914"/branch["selected_checkpoint_sha256"]).resolve():
            raise ValueError("Resume deployment path does not identify the newly selected large checkpoint")
        required = [deploy/file for file in ("float.pt", "model.bin", "model.calibration.json", "selection.json", "external.json", "clean.json", "primary.json")]
        frozen_hashes.update({str(path): sha(path) for path in required})
        model_sha, checkpoint_sha = sha(deploy/"model.bin"), sha(deploy/"float.pt")
        audit, selection = read(deploy/"model.calibration.json"), read(deploy/"selection.json")
        packed = load_gtcrn_integer(deploy/"model.bin", calibration=audit)
        if (packed.source_sha256 != checkpoint_sha or selection["checkpoint_sha256"] != checkpoint_sha
                or branch["model_sha256"] != model_sha or branch["selected_checkpoint_sha256"] != checkpoint_sha):
            raise ValueError("Large float checkpoint, deployment and selection hashes do not match")
        reports = {cohort: verified_report(deploy/(cohort+".json"), model_sha, cohort) for cohort in COHORTS}
        branches.append(dict(name=name, deployment=str(deploy), checkpoint_sha256=checkpoint_sha,
                             packed_sha256=model_sha, reports=reports))
    selected = choose_seed(branches)
    historical_dir = runs/"gtcrn_long_qat_batch16/candidates/epoch0001-ulylgdqg"
    if sha(historical_dir/"model.bin") != HISTORICAL_SHA:
        raise ValueError("Historical champion binary differs from the verified 9.2022 dB artifact")
    historical_path = runs/"gtcrn_long_qat_batch16/external_final_development.json"
    historical = verified_report(historical_path, HISTORICAL_SHA, "external")
    if historical["summary"]["si_sdri"] != HISTORICAL_SCORE:
        raise ValueError("Historical champion score changed")
    same_inputs(historical, selected["reports"]["external"])
    frozen_hashes.update({str(historical_dir/"model.bin"): HISTORICAL_SHA,
                          str(historical_path): sha(historical_path), **manifests})

    output = runs/"post_large_qat_20260914"
    output.mkdir(exist_ok=False)  # Never restart over candidates or failure evidence.
    inputs = output/"inputs"
    inputs.mkdir()
    deploy = Path(selected["deployment"])
    for name in ("float.pt", "model.bin", "model.calibration.json", "selection.json"):
        data = (deploy/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != frozen_hashes[str(deploy/name)]:
            raise ValueError("Selected input changed during freezing")
        (inputs/name).write_bytes(data)
        frozen_hashes[str(inputs/name)] = hashlib.sha256(data).hexdigest()
    parent = torch.load(inputs/"float.pt", map_location="cpu", weights_only=False)
    dataset, recipe = training_dataset(parent, COHORTS["external"][0])
    if len(dataset) < 4000 or parent["train_config"].get("waveform_loss_weight", .1) != .1:
        raise ValueError("Parent cannot supply the matched 4,000-example/.1-waveform-loss QAT recipe")
    for item in recipe["manifests"].values():
        frozen_hashes[item["path"]] = item["sha256"]
    config = dict(source_checkpoint=str(inputs/"float.pt"), integer_model=str(inputs/"model.bin"),
        calibration=str(inputs/"model.calibration.json"), development_manifest=COHORTS["external"][0],
        output_dir=str(output/"qat"), epochs=80, batch_size=16, workers=2, max_steps_per_epoch=250,
        max_hours=6, patience=20, device="cuda", learning_rate=3e-5, min_learning_rate=5e-6,
        waveform_loss_weight=.1, seed=20260914)
    QATTrainConfig(**config)
    write_once(output/"config.json", config)
    frozen_hashes[str(output/"config.json")] = sha(output/"config.json")
    selection = dict(seed={key: value for key, value in selected.items() if key != "reports"},
        branch_scores={branch["name"]: branch["reports"]["external"]["summary"] for branch in branches},
        selection_rule="Higher full C PCM16 external500 SI-SDRi among the two new large exports only",
        frozen_inputs_sha256=frozen_hashes, source_sha256=source_inventory,
        source_release_sha256=PINNED_RELEASE_SHA, runner_sha256=sha(__file__),
        training_recipe=recipe, bounds=dict(max_rounds=80, examples_per_round=4000, optimizer_steps_per_round=250,
            batch_size=16, training_hours=6, post_training_evaluation="additional bounded CPU time"),
        historical_champion=dict(packed_sha256=HISTORICAL_SHA, si_sdri=HISTORICAL_SCORE), official_test_used=False)
    write_once(output/"selection.json", selection)
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def verify():
        code_inventory(project)
        if any(sha(path) != expected for path, expected in frozen_hashes.items()):
            raise ValueError("A frozen model, report, source manifest or calibration changed")

    def run(command, log, timeout):
        verify()
        with log.open("x") as handle:
            subprocess.run(command, cwd=project, env=environment, stdout=handle,
                           stderr=subprocess.STDOUT, check=True, timeout=timeout)
        verify()

    try:
        print(json.dumps(dict(event="post_large_qat_started", seed=selected["name"], config=config)), flush=True)
        # Trainer's six-hour budget includes its repeated development selection.
        # Outer limit allows the in-progress selection to finish and save safely.
        run([sys.executable, "-u", "-m", "esp32_denoiser.gtcrn_qat_train", "--config", str(output/"config.json")],
            output/"training.log", 7*3600)
        summary, best = read(output/"qat/summary.json"), read(output/"qat/best.json")
        if summary.get("official_test_used") is not False or summary.get("best") != best:
            raise ValueError("QAT summary and immutable candidate selection disagree")
        candidate = Path(best["directory"]).resolve()
        if not candidate.is_relative_to((output/"qat/candidates").resolve()):
            raise ValueError("QAT selected a candidate outside this new experiment")
        if sha(candidate/"model.bin") != best["packed_sha256"] or sha(candidate/"model.pt") != best["checkpoint_sha256"]:
            raise ValueError("Selected QAT candidate hashes changed")
        frozen_hashes.update({str(candidate/name): sha(candidate/name) for name in ("model.bin", "model.pt", "model.calibration.json")})
        reports = {}
        for cohort, (manifest, _, _) in COHORTS.items():
            destination = output/(cohort+"_final_development.json")
            run([sys.executable, "-m", "esp32_denoiser.gtcrn_embedded", "--integer-model", str(candidate/"model.bin"),
                 "--calibration", str(candidate/"model.calibration.json"), "--manifest", manifest,
                 "--output", str(destination), "--io-format", "pcm16", "--perceptual", "--threads", "2"],
                output/(cohort+"_final_development.log"), 2*3600)
            reports[cohort] = verified_report(destination, best["packed_sha256"], cohort)
            same_inputs(selected["reports"][cohort], reports[cohort])
        comparison = compare_champion(historical, reports["external"])
        if not math.isclose(best["si_sdri"], reports["external"]["summary"]["si_sdri"], abs_tol=1e-10, rel_tol=0):
            raise ValueError("Selected QAT score differs from its full C replay")
        champion = global_champion([
            dict(kind="historical", directory=str(historical_dir), packed_sha256=HISTORICAL_SHA, report=historical),
            dict(kind="large_export", directory=selected["deployment"], packed_sha256=selected["packed_sha256"],
                 report=selected["reports"]["external"]),
            dict(kind="qat", directory=str(candidate), packed_sha256=best["packed_sha256"], report=reports["external"]),
        ])
        result = dict(status="complete", training_summary=summary, best=best,
            qat_comparison_to_historical=comparison,
            large_export_comparison_to_historical=compare_champion(historical, selected["reports"]["external"]),
            global_champion=champion, cohorts={key: value["summary"] for key, value in reports.items()},
            final_input_sha256=frozen_hashes,
            evaluation_sha256={cohort: sha(output/(cohort+"_final_development.json")) for cohort in COHORTS},
            official_test_used=False, scope="Development model selection; no final-test or ESP32 timing claim")
        verify()
        write_once(output/"champion_selection.json", champion)
        write_once(output/"complete.json", result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        write_once(output/"failure.json", dict(error=type(error).__name__, message=str(error), official_test_used=False))
        raise


if __name__ == "__main__":
    main()
