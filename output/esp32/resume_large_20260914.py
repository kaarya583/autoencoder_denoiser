"""Resume the two existing large runs, then freeze and evaluate their winners.

This coordinator changes no scientific source. Check-only is the default;
--execute starts work. A run-wide lock prevents duplicate coordinators. Child
jobs own new process sessions; cleanup verifies Linux PID birth identity.
Original files are copied before train.py replaces its mutable run metadata.
Five hours is an additional, persisted per-branch training allowance, not a
claim that either run will reach epoch 240. Completion requires all exports
and all three complete C/PCM16/perceptual evaluation reports.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile


BUNDLE_SHA = "eceb96444d0988d487f7a087190b5baf5437d54f949190dbf0ef67ec1e6a64d3"
RUNS = {"float_gtcrn_broad_large": 158, "float_gtcrn_broad_normalized_large": 151}
TAG = "resume_large_20260914"
TARGET = 240
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def records(path):
    text = Path(path).read_text()
    if text and not text.endswith("\n"):
        raise ValueError(f"Incomplete JSONL tail must be audited before resume: {path}")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def copy_immutable(source, destination):
    destination = Path(destination)
    if destination.exists():
        if sha(source) != sha(destination):
            raise ValueError(f"Immutable output differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp-" + uuid.uuid4().hex)
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def verify_source(project, bundle):
    if sha(bundle) != BUNDLE_SHA:
        raise ValueError("Source ZIP is not the reviewed eceb964 bundle")
    count = 0
    with zipfile.ZipFile(bundle) as archive:
        for entry in archive.infolist():
            relative = Path(entry.filename)
            if entry.is_dir() or relative.parts[0] not in {"esp32_denoiser", "firmware"}:
                continue
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe source ZIP member")
            target = project / relative
            if not target.is_file() or hashlib.sha256(archive.read(entry)).hexdigest() != sha(target):
                raise ValueError(f"Installed source differs from pinned bundle: {relative}")
            count += 1
    if count < 20:
        raise ValueError("Source bundle inventory is unexpectedly incomplete")
    return {"bundle_sha256": BUNDLE_SHA, "verified_installed_files": count}


def checkpoint(path):
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("phase") != "float" or saved.get("model_kind") != "gtcrn"
            or saved.get("initialization_only") or not saved.get("optimizer", {}).get("state")
            or not isinstance(saved.get("scheduler"), dict)):
        raise ValueError(f"Not a resumable trained float GTCRN checkpoint: {path}")
    return saved


def resume_config(saved, run, hours):
    """Copy the exact checkpoint recipe; only continuation controls change."""
    config = dict(saved["train_config"])
    config.update(output_dir=str(run), resume=str(run / "last.pt"), resume_optimizer=True,
                  epochs=TARGET, max_hours=hours)
    return config


def process_identity(pid):
    """Linux /proc start ticks plus boot and session IDs protect PID reuse."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": int(fields[19]), "pgid": int(fields[2]),
                "session_id": int(fields[3]),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "cmdline_sha256": sha(f"/proc/{pid}/cmdline")}
    except (FileNotFoundError, ProcessLookupError):
        return None


def identity_alive(record):
    return process_identity(record["pid"]) == record


def stop_owned(record):
    if record["pgid"] != record["pid"] or record["session_id"] != record["pid"]:
        raise ValueError("Refusing cleanup of a process without its own recorded session")
    for action, seconds in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        if not identity_alive(record):
            return
        try:
            os.killpg(record["pgid"], action)
        except ProcessLookupError:
            return
        end = time.monotonic() + seconds
        while identity_alive(record) and time.monotonic() < end:
            time.sleep(.1)
    if identity_alive(record):
        raise RuntimeError(f"Owned process did not stop: {record['pid']}")


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A resume coordinator is already active") from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}) + "\n")
        handle.flush()
        yield


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.cancel = threading.Event()
        self.guard = threading.Lock()
        self.jobs = []

    def spawn(self, command, directory, label):
        if self.cancel.is_set():
            raise RuntimeError("Coordinator cancellation requested")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (label + ".process.json")
        if path.exists():
            raise ValueError(f"A process record already exists: {path}")
        with (directory / (label + ".log")).open("w") as log:
            process = subprocess.Popen(command, cwd=self.args.project, env=ENV,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        identity = process_identity(process.pid)
        if identity is None:
            code = process.wait()
            raise RuntimeError(f"{label} exited before identity registration (status {code})")
        record = {"identity": identity, "command": command, "started_at": time.time(), "label": label}
        with self.guard:
            self.jobs.append((process, path, record))
        write_json(path, record)
        return process, path, record

    def poll(self, job):
        process, path, record = job
        code = process.poll()
        if code is not None and "returncode" not in record:
            record.update(returncode=code, ended_at=time.time())
            write_json(path, record)
        return code

    def wait(self, job, deadline, companion=None):
        while self.poll(job) is None:
            if self.cancel.is_set() or time.monotonic() >= deadline:
                raise TimeoutError(f"Cancelled or exceeded deadline: {job[2]['label']}")
            # The watcher sees summary.json just before train.py exits. A
            # successful watcher exit is therefore allowed while its trainer
            # is still alive; the exact final hash is checked again below.
            if companion is not None and self.poll(companion) not in (None, 0):
                raise RuntimeError("Development watcher failed during training")
            time.sleep(2)
        if self.poll(job) != 0:
            raise RuntimeError(f"{job[2]['label']} failed with status {self.poll(job)}")

    def command(self, command, directory, label, deadline):
        self.wait(self.spawn(command, directory, label), deadline)

    def cleanup(self):
        self.cancel.set()
        with self.guard:
            jobs = list(self.jobs)
        for job in jobs:
            if self.poll(job) is None:
                stop_owned(job[2]["identity"])
                job[0].wait(timeout=10)
                self.poll(job)


def preflight_run(run, minimum_epoch):
    saved = checkpoint(run / "last.pt")
    epoch = int(saved["epoch"])
    if not minimum_epoch <= epoch <= TARGET:
        raise ValueError(f"Unexpected checkpoint epoch {epoch}: {run}")
    config = saved["train_config"]
    provenance = saved["provenance"]
    for name in ("train", "val"):
        if sha(config[name + "_manifest"]) != provenance["manifest_sha256"][name]:
            raise ValueError("A saved training/validation manifest changed before resume")
    for name in ("speech", "noise"):
        if sha(config["synthetic_" + name + "_manifest"]) != provenance["added_training_sources"][name + "_manifest_sha256"]:
            raise ValueError("A saved synthetic source manifest changed before resume")
    if (config.get("epoch_samples") != 43208 or config.get("batch_size") != 48
            or config.get("max_steps_per_epoch") is not None or config.get("max_val_batches") is not None):
        raise ValueError(f"Not the existing full large-run recipe: {run}")
    normalized = "normalized" in run.name
    if bool(saved["model_config"].get("normalize_input", False)) != normalized:
        raise ValueError("Run name and input normalization differ")
    rows = records(run / "history.jsonl")
    if not rows or rows[-1]["epoch"] != epoch or len({row["epoch"] for row in rows}) != len(rows):
        raise ValueError("History is duplicated or does not match the durable checkpoint; audit before resuming")
    return {"run": str(run), "checkpoint_epoch": epoch, "last_checkpoint_sha256": sha(run / "last.pt"),
            "target_epochs": TARGET, "optimizer_learning_rates": [group["lr"] for group in saved["optimizer"]["param_groups"]],
            "stale_epochs": saved.get("stale", 0), "patience": config["patience"]}


def recover_owned_orphans(directory):
    """Only clean up jobs registered by a prior instance of this coordinator."""
    for path in directory.rglob("*.process.json"):
        record = read_json(path)
        if "returncode" not in record:
            if identity_alive(record["identity"]):
                stop_owned(record["identity"])
            record.update(recovered_at=time.time(), ended_at=time.time(),
                          returncode=None, outcome="prior coordinator interrupted; checkpoint resume required")
            write_json(path, record)


def assert_no_unowned_writers(run):
    """Never interrupt a trainer/selector started by another coordinator."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            arguments = (entry / "cmdline").read_bytes().decode().split("\0")
            target = None
            if "esp32_denoiser.train" in arguments and "--config" in arguments:
                config_path = arguments[arguments.index("--config") + 1]
                target = read_json(config_path).get("output_dir")
            elif "esp32_denoiser.development_checkpoints" in arguments and "--run" in arguments:
                target = arguments[arguments.index("--run") + 1]
            if target and Path(target).resolve() == run.resolve():
                raise RuntimeError(f"Existing writer PID {entry.name} targets {run}; it is not owned by this invocation")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue


def finished_stage(marker, binding):
    if not marker.exists():
        return False
    record = read_json(marker)
    if record["binding"] != binding:
        raise ValueError(f"Completed stage input identity changed: {marker}")
    for name, digest in record["artifacts"].items():
        if sha(marker.parent / name) != digest:
            raise ValueError(f"Completed artifact was modified: {name}")
    return True


def mark_stage(marker, binding, paths, **extra):
    write_json(marker, {"binding": binding, "artifacts": {path.name: sha(path) for path in paths}, **extra})


def validate_evaluation(report_path, manifest, expected):
    report = read_json(report_path)
    rows = records(manifest)
    ids = [row["id"] for row in rows]
    actual_ids = [row["id"] for row in report["utterances"]]
    summary = report["summary"]
    if (len(ids) != expected or len(set(ids)) != expected or len(actual_ids) != expected
            or set(ids) != set(actual_ids) or report["manifest_sha256"] != sha(manifest)
            or report["limited_evaluation"] or summary["utterances"] != expected
            or summary["valid_utterances"] != expected or summary["invalid_utterances"]):
        raise ValueError("Evaluation is not the complete finite, matching cohort")
    quality = report["perceptual"]["summary"]
    for metric in ("pesq_wb", "stoi"):
        values = quality[metric]
        if (values["valid_utterances"] < 1
                or values["valid_utterances"] + values["invalid_utterances"] != expected):
            raise ValueError("Perceptual evaluation has no valid pairs or incorrect accounting")
    return {"si_sdri": summary["si_sdri"], "perceptual": quality}


def branch(supervisor, name):
    args = supervisor.args
    run = args.root / name
    state = run / TAG
    state.mkdir(exist_ok=True)
    recover_owned_orphans(state)
    preflight_run(run, RUNS[name])
    plan_path = state / "plan.json"
    plan = {"training_allowance_hours": args.training_hours, "target_epochs": TARGET,
            "source_bundle_sha256": BUNDLE_SHA, "official_test_used": False}
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Persisted continuation plan differs; do not silently reset the allowance")
    write_json(plan_path, plan)
    finish = state / "training.finished.json"
    if not finish.exists():
        used = sum(max(0, record["ended_at"] - record["started_at"])
                   for path in state.rglob("training.process.json")
                   if "ended_at" in (record := read_json(path)))
        hours = max(0, args.training_hours - used / 3600)
        saved = checkpoint(run / "last.pt")
        if saved["epoch"] < TARGET and hours > 0:
            attempt = state / ("attempt_" + str(time.time_ns()))
            backup = attempt / "before"
            for relative in ("last.pt", "best.pt", "config.json", "large_config.json", "history.jsonl",
                             "provenance.json", "summary.json", "initial_validation.json", "best_validation.json",
                             "external_development/best.json", "external_development/history.jsonl"):
                source = run / relative
                if source.exists():
                    copy_immutable(source, backup / relative)
            if (run / "external_development/best.json").exists():
                selected = Path(read_json(run / "external_development/best.json")["selected_checkpoint"])
                copy_immutable(selected, backup / "external_development" / selected.name)
            config = resume_config(saved, run, hours)
            config_path = attempt / "resume_config.json"
            write_json(config_path, config)
            # Existing summary/provenance/config bytes remain in the backup.
            # Only summary must disappear so the watcher does not exit early.
            (run / "summary.json").unlink(missing_ok=True)
            training = supervisor.spawn([sys.executable, "-u", "-m", "esp32_denoiser.train",
                                         "--config", str(config_path)], attempt, "training")
            watcher = supervisor.spawn([sys.executable, "-u", "-m", "esp32_denoiser.development_checkpoints",
                                        "--run", str(run), "--manifest", str(args.external), "--device", "cuda",
                                        "--every-epochs", "10", "--max-hours", str(hours + .5)], attempt, "selector")
            print(json.dumps({"event": "resume_started", "run": name, "from_epoch": saved["epoch"],
                              "additional_training_hours": hours, "optimizer_preserved": True,
                              "training_pid": training[0].pid}), flush=True)
            supervisor.wait(training, time.monotonic() + hours * 3600 + 1200, companion=watcher)
            supervisor.wait(watcher, time.monotonic() + 1200)
            saved = checkpoint(run / "last.pt")
            summary = read_json(run / "summary.json")
            if summary.get("epoch", saved["epoch"]) != saved["epoch"]:
                raise ValueError("Training summary/checkpoint epoch mismatch")
            status = summary["status"]
        else:
            status = "target_already_reached" if saved["epoch"] == TARGET else "coordinator_training_allowance_exhausted"
        preflight_run(run, RUNS[name])
        write_json(finish, {"last_checkpoint_sha256": sha(run / "last.pt"), "final_epoch": saved["epoch"],
                            "training_status": status, "target_reached": saved["epoch"] == TARGET})
    training_result = read_json(finish)
    if training_result["last_checkpoint_sha256"] != sha(run / "last.pt"):
        raise ValueError("Completed training checkpoint was changed after this coordinator's run")
    evaluation_end = time.monotonic() + args.evaluation_hours * 3600
    stage_logs = state / ("evaluation_" + str(time.time_ns()))
    history = run / "external_development/history.jsonl"
    scored = records(history) if history.exists() else []
    if not any(row.get("checkpoint_sha256") == training_result["last_checkpoint_sha256"] for row in scored):
        script = ("from pathlib import Path; from esp32_denoiser.development_checkpoints import score_checkpoint; "
                  "import sys; score_checkpoint(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),device='cuda')")
        supervisor.command([sys.executable, "-u", "-c", script, str(run / "last.pt"), str(args.external),
                            str(run / "external_development")], stage_logs, "final_selection", evaluation_end)
    winner_path = run / "external_development/best.json"
    winner = read_json(winner_path)
    selected = Path(winner["selected_checkpoint"])
    if (sha(selected) != winner["checkpoint_sha256"] or winner["manifest_sha256"] != sha(args.external)
            or winner["model"]["precision"] != "float32 PyTorch"
            or winner["validation"]["total_utterances"] != 500 or winner["validation"]["invalid_utterances"]):
        raise ValueError("External winner identity or complete development cohort is invalid")
    deploy = run / "deployment_resume_20260914" / winner["checkpoint_sha256"]
    deploy.mkdir(parents=True, exist_ok=True)
    copy_immutable(selected, deploy / "float.pt")
    copy_immutable(winner_path, deploy / "selection.json")
    binding = {"checkpoint_sha256": winner["checkpoint_sha256"], "source_bundle_sha256": BUNDLE_SHA,
               "calibration_crops": 128, "calibration_seed": 483, "external_manifest_sha256": sha(args.external)}
    calibration_marker = deploy / "calibration.complete.json"
    if not finished_stage(calibration_marker, binding):
        supervisor.command([sys.executable, "-u", "-m", "esp32_denoiser.gtcrn_mse_calibration",
                            "--checkpoint", str(deploy / "float.pt"), "--manifest", str(args.external),
                            "--output", str(deploy / "model.bin"), "--calibration-crops", "128",
                            "--calibration-seed", "483", "--threads", "2"], stage_logs, "calibration", evaluation_end)
        audit = read_json(deploy / "model.calibration.json")
        if audit["source_checkpoint_sha256"] != winner["checkpoint_sha256"] or not 0 < (deploy / "model.bin").stat().st_size <= 99000:
            raise ValueError("Packed model exceeds budget or calibration checkpoint differs")
        mark_stage(calibration_marker, binding, [deploy / name for name in ("model.bin", "model.calibration.json", "model.json")])
    evaluation_binding = {**binding, "model_sha256": sha(deploy / "model.bin"), "io_format": "pcm16", "perceptual": True}
    summaries = {}
    for cohort, manifest, count in (("external", args.external, 500), ("clean", args.clean, 100), ("primary", args.primary, 770)):
        bound = {**evaluation_binding, "manifest_sha256": sha(manifest), "cohort": cohort}
        marker = deploy / (cohort + ".complete.json")
        report = deploy / (cohort + ".json")
        if not finished_stage(marker, bound):
            supervisor.command([sys.executable, "-u", "-m", "esp32_denoiser.gtcrn_embedded",
                                "--integer-model", str(deploy / "model.bin"), "--calibration", str(deploy / "model.calibration.json"),
                                "--manifest", str(manifest), "--output", str(report), "--io-format", "pcm16",
                                "--perceptual", "--threads", "2"], stage_logs, cohort, evaluation_end)
            metrics = validate_evaluation(report, manifest, count)
            mark_stage(marker, bound, [report], metrics=metrics)
        summaries[cohort] = validate_evaluation(report, manifest, count)
    result = {"run": name, **training_result, "deployment": str(deploy), "model_bytes": (deploy / "model.bin").stat().st_size,
              "selected_checkpoint_sha256": winner["checkpoint_sha256"], "selected_epoch": winner["model"]["checkpoint_epoch"],
              "model_sha256": sha(deploy / "model.bin"), "evaluations": summaries,
              "pipeline_complete": True, "official_test_used": False, "hardware_tested": False}
    write_json(state / "complete.json", result)
    print(json.dumps({"event": "resume_branch_complete", "run": name, "target_reached": result["target_reached"],
                      "final_epoch": result["final_epoch"], "model_bytes": result["model_bytes"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path("/content/esp32_project"))
    parser.add_argument("--root", type=Path, default=Path("/content/esp32_runs"))
    parser.add_argument("--external", type=Path, default=Path("/content/extra_audio/development/mixtures.jsonl"))
    parser.add_argument("--clean", type=Path, default=Path("/content/extra_audio/development/clean.jsonl"))
    parser.add_argument("--primary", type=Path, default=Path("/content/voicebank/manifests/val.jsonl"))
    parser.add_argument("--training-hours", type=float, default=5)
    parser.add_argument("--evaluation-hours", type=float, default=2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if any(not math.isfinite(value) or value <= 0 for value in (args.training_hours, args.evaluation_hours)):
        parser.error("Time allowances must be positive and finite")
    args.project, args.root = args.project.resolve(), args.root.resolve()
    source = verify_source(args.project, args.source_bundle)
    for manifest, expected in ((args.external, 500), (args.clean, 100), (args.primary, 770)):
        if len(records(manifest)) != expected:
            raise ValueError(f"Unexpected development cohort size: {manifest}")
    plans = [preflight_run(args.root / name, minimum) for name, minimum in RUNS.items()]
    print(json.dumps({"event": "preflight_passed", "source": source, "branches": plans,
                      "execute": args.execute, "training_hours_per_branch": args.training_hours}), flush=True)
    if not args.execute:
        return
    if not Path("/proc/sys/kernel/random/boot_id").is_file():
        raise RuntimeError("Execution requires Linux process identity support")
    coordinator = args.root / TAG
    with exclusive(coordinator / "coordinator.lock"):
        for name in RUNS:
            recover_owned_orphans(args.root / name / TAG)
            assert_no_unowned_writers(args.root / name)
        supervisor = Supervisor(args)
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: supervisor.cancel.set())
        write_json(coordinator / "coordinator.json", {"identity": process_identity(os.getpid()),
                   "script_sha256": sha(__file__), "source": source, "started_at": time.time()})
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(branch, supervisor, name) for name in RUNS]
                # Workers cancel siblings immediately, before executor waits.
                for future in futures:
                    future.add_done_callback(lambda completed: supervisor.cancel.set() if completed.exception() else None)
                results = [future.result() for future in futures]
            write_json(coordinator / "complete.json", {"pipeline_complete": True, "branches": results,
                       "all_epoch_targets_reached": all(result["target_reached"] for result in results),
                       "official_test_used": False, "source_bundle_sha256": BUNDLE_SHA})
        except BaseException as error:
            write_json(coordinator / ("failure_" + str(time.time_ns()) + ".json"),
                       {"error": repr(error), "time": time.time(), "pipeline_complete": False})
            raise
        finally:
            supervisor.cleanup()


if __name__ == "__main__":
    main()
