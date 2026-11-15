"""Resumable actor-only QAT distillation on authorized simulated observations.

The float32 teacher remains frozen. Validation uses separate simulated matches
from the training seed range; final release deals are never sampled here.
"""

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import random
import shutil
import signal
import sys
import time
from pathlib import Path

import torch

from .artifacts import digest, read_quantized
from .bundle import training_commit
from .curriculum import atomic_json
from .export import serialize
from .model import ENCODER_VERSION, Policy, batch_distribution
from .population import Population
from .quantization import distribution_metrics, policy_kl, quantization_aware
from .resources import MemoryBudget
from .rollout import SimulatorPool, collect
from .snapshot import freeze_source, training_snapshot, verify_snapshot


def save(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=714)
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--validation-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument(
        "--updates",
        type=int,
        default=2000,
        help="maximum additional optimizer updates this invocation",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=1,
        help="maximum additional running hours this invocation",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--environments", type=int, default=64)
    parser.add_argument("--memory-gib", type=float, default=16)
    args = parser.parse_args()

    def interrupted(*_):
        raise KeyboardInterrupt("QAT interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    if any(
        value <= 0
        for value in (
            args.steps,
            args.validation_steps,
            args.batch_size,
            args.validate_every,
            args.updates,
        )
    ):
        parser.error("positive sizes and update counts required")
    if (
        not math.isfinite(args.hours)
        or args.hours <= 0
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error("positive finite hours and learning rate required")
    if (
        not 1 <= args.workers <= 64
        or not 1 <= args.environments <= 64
        or not 0 < args.memory_gib <= 16
    ):
        parser.error("invalid simulation or memory bounds")
    if args.smoke and (
        args.steps > 256
        or args.validation_steps > 256
        or args.updates > 10
        or args.hours > 0.01
    ):
        parser.error(
            "smoke jobs are limited to 256 observations per bank, 10 updates and .01 hours per invocation"
        )
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "qat.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("this QAT directory already has a live trainer")
    teacher_path = args.teacher.resolve()
    source = root / "source"
    repository = Path(
        os.environ.get("RETRUCO_ML_REPOSITORY", Path(__file__).resolve().parents[2])
    )
    if Path(__file__).resolve().parent != source / "ml/retruco_ml":
        freeze_source(source, repository, allow_dirty=args.smoke)
        # Re-exec from the snapshot. CWD must also change: otherwise Python's
        # module search would prefer the editable working tree over PYTHONPATH.
        command = [
            sys.executable,
            "-u",
            "-m",
            "retruco_ml.qat",
            str(teacher_path),
            "--output",
            str(root),
        ]
        for name in (
            "seed",
            "steps",
            "validation_steps",
            "batch_size",
            "learning_rate",
            "validate_every",
            "updates",
            "hours",
            "workers",
            "environments",
            "memory_gib",
        ):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        for name in ("resume", "smoke"):
            if getattr(args, name):
                command.append("--" + name)
        environment = dict(
            os.environ,
            PYTHONPATH=str(source / "ml"),
            RETRUCO_ML_REPOSITORY=str(repository),
        )
        os.chdir(source / "ml")
        os.execve(sys.executable, command, environment)
    snapshot = verify_snapshot(source)
    if not args.smoke and snapshot.get("source_dirty") is not False:
        raise ValueError("cannot release from a dirty training snapshot")
    checkpoint = root / "checkpoint.pt"
    if checkpoint.exists() and not args.resume:
        raise ValueError("QAT checkpoint exists; use --resume or another directory")
    if args.resume and not checkpoint.exists():
        raise ValueError("there is no QAT checkpoint to resume")
    teacher_bytes = teacher_path.read_bytes()
    teacher_hash = hashlib.sha256(teacher_bytes).hexdigest()
    teacher_saved = torch.load(
        io.BytesIO(teacher_bytes), map_location="cpu", weights_only=True
    )
    if (
        teacher_saved.get("encoder") != ENCODER_VERSION
        or teacher_saved.get("training_kind", "ppo") != "ppo"
    ):
        raise ValueError(
            "QAT requires a compatible float32 PPO teacher; resume existing QAT runs directly"
        )
    teacher_manifest = training_snapshot(teacher_path)
    origin = {
        "teacher_checkpoint_sha256": teacher_hash,
        "teacher_training_snapshot_sha256": digest(teacher_manifest),
        "teacher_training_source_commit": training_commit(teacher_manifest, repository),
    }
    configuration = {
        **origin,
        "seed": args.seed,
        "steps": args.steps,
        "validation_steps": args.validation_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "validate_every": args.validate_every,
        "workers": args.workers,
        "environments": args.environments,
        "memory_gib": args.memory_gib,
        "source_manifest_sha256": digest(source / "manifest.json"),
        "smoke": args.smoke,
    }
    if (root / "configuration.json").exists() and json.loads(
        (root / "configuration.json").read_text()
    ) != configuration:
        raise ValueError("QAT teacher, calibration or optimizer settings changed")
    atomic_json(root / "configuration.json", configuration)
    if (root / "teacher.pt").exists() and digest(root / "teacher.pt") != teacher_hash:
        raise ValueError("frozen teacher changed")
    (root / "teacher.pt").write_bytes(teacher_bytes)
    shutil.copy2(teacher_manifest, root / "teacher-source-manifest.json")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    teacher = Policy()
    teacher.load_state_dict(teacher_saved["model"])
    teacher.eval()
    student = quantization_aware(teacher)
    optimizer = torch.optim.Adam(
        [p for p in student.parameters() if p.requires_grad], lr=args.learning_rate
    )
    rng = random.Random(args.seed + 2)
    status = {
        "pid": os.getpid(),
        "status": "running",
        "stage": "calibration",
        "started": time.time(),
    }
    atomic_json(root / "status.json", status)
    try:
        with SimulatorPool(args.workers) as pool:
            monitor = MemoryBudget(
                [s.process for s in pool.simulators], args.memory_gib
            )
            monitor.check(force=True)
            bank_path = root / "calibration.pt"
            if not bank_path.exists():
                population = Population(teacher_saved.get("population", []))
                train, _, _ = collect(
                    teacher,
                    pool,
                    population,
                    random.Random(args.seed),
                    args.steps,
                    players=0,
                    environments=args.environments,
                    monitor=monitor,
                )
                validation, _, _ = collect(
                    teacher,
                    pool,
                    population,
                    random.Random(args.seed + 1),
                    args.validation_steps,
                    players=0,
                    environments=args.environments,
                    monitor=monitor,
                )
                save(
                    bank_path,
                    {
                        "teacher_sha256": teacher_hash,
                        "encoder": ENCODER_VERSION,
                        "train": [r[0] for r in train],
                        "validation": [r[0] for r in validation],
                    },
                )
                del train, validation
            bank = torch.load(bank_path, map_location="cpu", weights_only=True)
            if (
                bank["teacher_sha256"] != teacher_hash
                or bank["encoder"] != ENCODER_VERSION
            ):
                raise ValueError("calibration identity mismatch")
            bank_hash = digest(bank_path)
            train = bank["train"]
            validation = bank["validation"]
            if len(train) < args.steps or len(validation) < args.validation_steps:
                raise ValueError("incomplete calibration bank")
            baseline = distribution_metrics(
                teacher, read_quantized(serialize(student)[0], teacher), validation
            )
            best = baseline
            completed = 0
            previous_seconds = 0.0
            best_update = 0
            best_model = {
                k: v.detach().clone() for k, v in student.state_dict().items()
            }
            if args.resume:
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if saved.get("evaluation_only"):
                    raise ValueError(
                        "resume checkpoint.pt; best.pt is an evaluation artifact"
                    )
                if (
                    saved["qat"]["configuration"] != configuration
                    or saved["qat"]["calibration_sha256"] != bank_hash
                ):
                    raise ValueError("QAT checkpoint inputs changed")
                student.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                rng.setstate(saved["random"])
                torch.set_rng_state(saved["torch_random"])
                completed = saved["update"]
                previous_seconds = saved["qat"]["elapsed_seconds"]
                baseline = saved["qat"]["baseline"]
                best = saved["qat"]["best_validation"]
                best_model = saved["best_model"]
                best_update = saved["best_update"]
            started = time.monotonic()
            last_save = started

            def saved_state(update):
                elapsed = previous_seconds + time.monotonic() - started
                return {
                    "encoder": ENCODER_VERSION,
                    "training_kind": "qat",
                    "model": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "random": rng.getstate(),
                    "torch_random": torch.get_rng_state(),
                    "best_model": best_model,
                    "best_update": best_update,
                    "update": update,
                    "total_seconds": teacher_saved.get("total_seconds", 0) + elapsed,
                    "initial_seed": teacher_saved.get(
                        "initial_seed", teacher_saved.get("config", {}).get("seed")
                    ),
                    "config": configuration,
                    "qat": {
                        **origin,
                        "configuration": configuration,
                        "calibration_sha256": bank_hash,
                        "baseline": baseline,
                        "best_validation": best,
                        "elapsed_seconds": elapsed,
                        "teacher_ppo_update": teacher_saved.get("update"),
                        "seed_range": "training-only, >=2^32",
                    },
                }

            def save_selected(value):
                # Keep the best weights inside the same atomic resume checkpoint.
                # Export a smaller evaluation-only artifact; it cannot accidentally
                # pair the chosen weights with a later optimizer state on resume.
                selected = {
                    k: v
                    for k, v in value.items()
                    if k
                    not in (
                        "optimizer",
                        "random",
                        "torch_random",
                        "best_model",
                        "best_update",
                    )
                }
                selected.update(
                    model=best_model, update=best_update, evaluation_only=True
                )
                save(root / "best.pt", selected)

            if not args.resume:
                value = saved_state(0)
                save(checkpoint, value)
                save_selected(value)
            for update in range(completed + 1, completed + args.updates + 1):
                monitor.check()
                batch = [
                    train[rng.randrange(len(train))] for _ in range(args.batch_size)
                ]
                with torch.no_grad():
                    reference, _ = batch_distribution(teacher, batch)
                predicted, _ = batch_distribution(student, batch)
                loss = policy_kl(reference, predicted)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite QAT loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
                optimizer.step()
                finished = (
                    update == completed + args.updates
                    or time.monotonic() - started >= args.hours * 3600
                )
                if (
                    update % args.validate_every == 0
                    or finished
                    or time.monotonic() - last_save >= 900
                ):
                    # Score the actual serialized/dequantized actor, not just its
                    # training proxy, before selecting a candidate checkpoint.
                    deployed = read_quantized(serialize(student)[0], teacher)
                    measured = distribution_metrics(teacher, deployed, validation)
                    improved = (
                        measured["mean_kl"] < best["mean_kl"]
                        and measured["p95_total_variation"]
                        <= baseline["p95_total_variation"]
                    )
                    if improved:
                        best = measured
                        best_update = update
                        best_model = {
                            k: v.detach().clone()
                            for k, v in student.state_dict().items()
                        }
                    value = saved_state(update)
                    save(checkpoint, value)
                    save_selected(value)
                    metric = {
                        "update": update,
                        "loss": float(loss.detach()),
                        "validation": measured,
                        "selected": improved,
                        "elapsed_seconds": value["qat"]["elapsed_seconds"],
                        "peak_group_rss_bytes": monitor.peak,
                    }
                    with (root / "metrics.jsonl").open("a") as log:
                        log.write(json.dumps(metric) + "\n")
                    print(json.dumps(metric), flush=True)
                    status.update(stage="optimization", update=update)
                    atomic_json(root / "status.json", status)
                    last_save = time.monotonic()
                if finished:
                    break
            selected = torch.load(
                root / "best.pt", map_location="cpu", weights_only=True
            )
            atomic_json(
                root / "candidate.json",
                {
                    "approved": False,
                    "checkpoint": str(root / "best.pt"),
                    "checkpoint_sha256": digest(root / "best.pt"),
                    "teacher_checkpoint_sha256": teacher_hash,
                    "baseline": baseline,
                    "validation": selected["qat"]["best_validation"],
                    "warning": "Distribution preservation is not playing-strength or release approval evidence.",
                },
            )
        status.update(
            status="complete", stage="candidate-ready-for-strength-assessment"
        )
    except BaseException as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        atomic_json(root / "status.json", status)


if __name__ == "__main__":
    main()
