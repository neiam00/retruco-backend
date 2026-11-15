"""Frozen-source assessment campaign. Produces evidence, never release approval."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

from .artifacts import digest, validate_manifest
from .assessment import load_checkpoint
from .curriculum import atomic_json
from .evaluation_styles import MIXTURE_STYLES
from .export import serialize
from .holdout import reserve
from .model import ENCODER_VERSION, Policy
from .snapshot import freeze_source, verify_snapshot
from .splits import VALIDATION_START


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def freeze_inputs(args, root):
    """Copy each checkpoint once; resume keeps the original inputs, even if an
    external training process subsequently replaces its checkpoint path."""
    folder = root / "inputs"
    manifest_path = folder / "manifest.json"
    requested = {
        "checkpoint": str(args.checkpoint),
        "bundle": str(args.bundle),
        "partners": [str(p) for p in args.partner_checkpoint],
        "historical": [str(p) for p in args.historical_checkpoint],
        "teacher": str(args.teacher_checkpoint) if args.teacher_checkpoint else None,
    }
    if manifest_path.exists():
        metadata = json.loads(manifest_path.read_text())
        if metadata["requested"] != requested:
            raise ValueError("input paths changed; use a new campaign directory")
        for name, expected in metadata["files"].items():
            if digest(root / name) != expected:
                raise ValueError("frozen assessment input changed: " + name)
        return metadata
    folder.mkdir(exist_ok=True)
    files = {}

    def copy(path, name):
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        target.write_bytes(data)
        relative = str(target.relative_to(root))
        files[relative] = hashlib.sha256(data).hexdigest()
        return relative

    checkpoint = copy(args.checkpoint, "candidate.pt")
    model = load_checkpoint(root / checkpoint)
    saved = torch.load(root / checkpoint, map_location="cpu", weights_only=True)
    candidate_actor = hashlib.sha256(serialize(model)[0]).hexdigest()
    for name in (
        "manifest.json",
        "policy.bin",
        "policy.wasm",
        "reference.f32",
        "training-source-manifest.json",
    ):
        copy(args.bundle / name, "bundle/" + name)
    bundle = folder / "bundle"
    manifest = json.loads((bundle / "manifest.json").read_text())
    validate_manifest(
        manifest,
        (bundle / "policy.bin").read_bytes(),
        (bundle / "policy.wasm").read_bytes(),
    )
    exported, reference = serialize(model)
    if (
        manifest["checkpoint_sha256"] != model.checkpoint_sha256
        or (bundle / "policy.bin").read_bytes() != exported
        or (bundle / "reference.f32").read_bytes() != reference
    ):
        raise ValueError("checkpoint and bundle differ")
    if not args.mode == "smoke" and manifest.get("source_dirty") is not False:
        raise ValueError("assessment requires a clean candidate bundle")
    candidate_seed = saved.get("initial_seed", saved.get("config", {}).get("seed"))
    teacher = None
    history_source = saved
    if model.training_kind == "qat":
        teacher = copy(
            args.teacher_checkpoint or args.checkpoint.parent / "teacher.pt",
            "teacher.pt",
        )
        if digest(root / teacher) != model.qat_origin["teacher_checkpoint_sha256"]:
            raise ValueError("QAT teacher identity mismatch")
        history_source = torch.load(
            root / teacher, map_location="cpu", weights_only=True
        )
        copy(
            args.bundle / "teacher-training-source-manifest.json",
            "bundle/teacher-training-source-manifest.json",
        )
    partners = []
    partner_seeds = set()
    partner_actors = set()
    for index, path in enumerate(args.partner_checkpoint):
        name = copy(path, f"partner-{index}.pt")
        other = load_checkpoint(root / name)
        state = torch.load(root / name, map_location="cpu", weights_only=True)
        seed = state.get("initial_seed", state.get("config", {}).get("seed"))
        actor = hashlib.sha256(serialize(other)[0]).hexdigest()
        if (
            type(seed) is not int
            or type(candidate_seed) is not int
            or seed == candidate_seed
            or seed in partner_seeds
            or actor == candidate_actor
            or actor in partner_actors
        ):
            raise ValueError(
                "partners must have distinct independent seeds and actor weights"
            )
        partner_seeds.add(seed)
        partner_actors.add(actor)
        partners.append(name)
    if len(partners) < 2:
        raise ValueError(
            "at least two independently trained partner checkpoints required"
        )
    historical = []
    actors = set()
    if args.historical_checkpoint:
        for index, path in enumerate(args.historical_checkpoint):
            name = copy(path, f"historical-{index}.pt")
            other = load_checkpoint(root / name)
            actor = hashlib.sha256(serialize(other)[0]).hexdigest()
            if actor == candidate_actor or actor in actors:
                raise ValueError(
                    "historical actors must differ from candidate and each other"
                )
            historical.append(name)
            actors.add(actor)
    else:
        choices = []
        for state in history_source.get("population", []):
            other = Policy()
            other.load_state_dict(state)
            actor = hashlib.sha256(serialize(other)[0]).hexdigest()
            if actor != candidate_actor and actor not in actors:
                choices.append(state)
                actors.add(actor)
        if len(choices) >= 2:
            for index, state in enumerate((choices[0], choices[-1])):
                path = folder / f"historical-{index}.pt"
                torch.save(
                    {
                        "encoder": ENCODER_VERSION,
                        "model": state,
                        "config": {"seed": candidate_seed},
                        "origin_checkpoint_sha256": digest(root / teacher)
                        if teacher
                        else model.checkpoint_sha256,
                    },
                    path,
                )
                name = str(path.relative_to(root))
                historical.append(name)
                files[name] = digest(path)
    if len(historical) < 2:
        raise ValueError("at least two distinct historical actors required")
    metadata = {
        "requested": requested,
        "files": files,
        "checkpoint": checkpoint,
        "bundle": str(bundle.relative_to(root)),
        "partners": partners,
        "historical": historical,
        "teacher": teacher,
        "checkpoint_sha256": model.checkpoint_sha256,
        "model_sha256": manifest["sha256"],
        "wasm_sha256": manifest["wasm_sha256"],
    }
    atomic_json(manifest_path, metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partner-checkpoint", type=Path, action="append", default=[])
    parser.add_argument(
        "--historical-checkpoint", type=Path, action="append", default=[]
    )
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument(
        "--mode", choices=("smoke", "validation", "release"), default="release"
    )
    parser.add_argument("--pairs", type=int)
    parser.add_argument("--bluff-pairs", type=int)
    parser.add_argument("--exploiter-hours", type=float, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--environments", type=int, default=64)
    parser.add_argument("--verification", type=Path)
    args = parser.parse_args()
    if args.pairs is None:
        args.pairs = {"smoke": 2, "validation": 128, "release": 2000}[args.mode]
    if args.bluff_pairs is None:
        args.bluff_pairs = {"smoke": 2, "validation": 128, "release": 1000}[args.mode]
    if (
        args.pairs < 2
        or args.bluff_pairs < 2
        or not 1 <= args.workers <= 64
        or not 1 <= args.environments <= 64
    ):
        parser.error("invalid pair counts or simulation bounds")
    if not math.isfinite(args.exploiter_hours) or args.exploiter_hours < 1:
        parser.error(
            "release exploiters require at least one hour per independent seed"
        )
    if args.mode == "release" and (args.pairs < 2000 or args.bluff_pairs < 1000):
        parser.error("release sample counts are below the required minimum")
    if args.mode == "smoke" and (args.pairs > 8 or args.bluff_pairs > 8):
        parser.error("smoke assessment is limited to eight pairs")
    for name in (
        "checkpoint",
        "bundle",
        "output",
        "teacher_checkpoint",
        "verification",
    ):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    args.partner_checkpoint = [p.resolve() for p in args.partner_checkpoint]
    args.historical_checkpoint = [p.resolve() for p in args.historical_checkpoint]
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "evaluation.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("this evaluation campaign already has a live supervisor")

    def interrupted(*_):
        raise KeyboardInterrupt("evaluation campaign interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    repository = Path(
        os.environ.get("RETRUCO_ML_REPOSITORY", Path(__file__).resolve().parents[2])
    )
    source = root / "source"
    if Path(__file__).resolve().parent != source / "ml/retruco_ml":
        freeze_source(source, repository, allow_dirty=args.mode == "smoke")
        command = [
            sys.executable,
            "-u",
            "-m",
            "retruco_ml.evaluation_campaign",
            str(args.checkpoint),
            str(args.bundle),
            "--output",
            str(root),
        ]
        for name in (
            "mode",
            "pairs",
            "bluff_pairs",
            "exploiter_hours",
            "workers",
            "environments",
        ):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        for name in ("partner_checkpoint", "historical_checkpoint"):
            for path in getattr(args, name):
                command.extend(["--" + name.replace("_", "-"), str(path)])
        for name in ("teacher_checkpoint", "verification"):
            if getattr(args, name):
                command.extend(
                    ["--" + name.replace("_", "-"), str(getattr(args, name))]
                )
        environment = dict(
            os.environ,
            PYTHONPATH=str(source / "ml"),
            RETRUCO_ML_REPOSITORY=str(repository),
        )
        os.chdir(source / "ml")
        os.execve(sys.executable, command, environment)
    snapshot = verify_snapshot(source)
    if args.mode != "smoke" and snapshot.get("source_dirty") is not False:
        raise ValueError("non-smoke assessment requires committed source")
    inputs = freeze_inputs(args, root)
    configuration = {
        "mode": args.mode,
        "pairs": args.pairs,
        "bluff_pairs": args.bluff_pairs,
        "exploiter_hours": args.exploiter_hours,
        "workers": args.workers,
        "environments": args.environments,
        "inputs_sha256": digest(root / "inputs/manifest.json"),
        "source_manifest_sha256": digest(source / "manifest.json"),
    }
    configuration_path = root / "configuration.json"
    if (
        configuration_path.exists()
        and json.loads(configuration_path.read_text()) != configuration
    ):
        raise ValueError("campaign settings changed; use a new directory")
    atomic_json(configuration_path, configuration)
    fingerprint = canonical_hash(configuration)
    reservation = (
        reserve(
            repository / "ml/runs/release-seeds.json",
            fingerprint,
            args.pairs,
            args.bluff_pairs,
        )
        if args.mode == "release"
        else {
            "seed": VALIDATION_START if args.mode == "validation" else 230000,
            "bluff_seed": 150000 if args.mode == "validation" else 230000,
            "pairs": args.pairs,
            "bluff_pairs": args.bluff_pairs,
            "fingerprint": fingerprint,
        }
    )
    atomic_json(root / "seed-reservation.json", reservation)
    for folder in ("reports", "logs", "receipts"):
        (root / folder).mkdir(exist_ok=True)
    checkpoint = root / inputs["checkpoint"]
    bundle = root / inputs["bundle"]
    model_hash = inputs["model_sha256"]
    status = {
        "pid": os.getpid(),
        "status": "running",
        "started": time.time(),
        "mode": args.mode,
    }
    environment = dict(os.environ, PYTHONPATH=str(source / "ml"))

    def run(name, command, output, dependencies=None):
        verify_snapshot(source)
        for path, expected in inputs["files"].items():
            if digest(root / path) != expected:
                raise ValueError("assessment inputs changed during campaign")
        dependencies = {str(path): digest(path) for path in (dependencies or ())}
        receipt = root / "receipts" / f"{name}.json"
        log = root / "logs" / f"{name}.log"
        prior = json.loads(receipt.read_text()) if receipt.exists() else None
        if (
            prior
            and prior["command"] == command
            and prior.get("dependencies", {}) == dependencies
            and prior["exit_code"] == 0
            and output.exists()
            and digest(output) == prior["output_sha256"]
            and log.exists()
            and digest(log) == prior["log_sha256"]
        ):
            return
        print(json.dumps({"stage": name, "mode": args.mode}), flush=True)
        with log.open("a") as stream:
            child = subprocess.Popen(
                [sys.executable, "-u", "-m", *command],
                cwd=source / "ml",
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            status.update(stage=name, child_pid=child.pid)
            atomic_json(root / "status.json", status)
            try:
                code = child.wait()
            except BaseException:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
        if code == 0 and output.suffix == ".json":
            value = json.loads(output.read_text())
            value.update(
                evaluation_source_commit=snapshot["source_commit"],
                evaluation_source_manifest_sha256=configuration[
                    "source_manifest_sha256"
                ],
                evaluation_fingerprint=fingerprint,
            )
            atomic_json(output, value)
        entry = {
            "command": command,
            "dependencies": dependencies,
            "exit_code": code,
            "log_sha256": digest(log),
            "output_sha256": digest(output) if output.exists() else None,
        }
        atomic_json(receipt, entry)
        if code or not output.exists():
            raise RuntimeError(f"{name} failed; inspect {log}")

    def assess(name, reference=False, players=0, extra=(), candidate=checkpoint):
        output = root / "reports" / f"{name}.json"
        command = [
            "retruco_ml.assessment",
            str(candidate),
            "--pairs",
            str(args.pairs),
            "--seed",
            str(reservation["seed"]),
            "--players",
            str(players),
            "--workers",
            str(args.workers),
            "--environments",
            str(args.environments),
            "--output",
            str(output),
            *extra,
        ]
        if not reference:
            command.extend(["--quantized", str(bundle / "policy.bin")])
        dependencies = [candidate]
        if not reference:
            dependencies.append(bundle / "policy.bin")
        for index, value in enumerate(extra):
            if value in (
                "--opponent-checkpoint",
                "--partner-checkpoint",
                "--compare-reference",
            ):
                dependencies.append(Path(extra[index + 1]))
        run(name, command, output, dependencies)
        report = json.loads(output.read_text())
        if (
            report["checkpoint_sha256"] != digest(candidate)
            or report["seed"] != reservation["seed"]
            or any(c["pairs"] != args.pairs for c in report["configurations"])
        ):
            raise ValueError("assessment output identity mismatch")
        return str(output.relative_to(root))

    evidence = {"checkpoint": inputs["checkpoint"], "bundle": inputs["bundle"]}
    try:
        evidence["reference"] = assess("reference", reference=True)
        evidence["quantized"] = assess(
            "quantized",
            extra=("--compare-reference", str(root / evidence["reference"])),
        )
        if inputs["teacher"]:
            evidence["qat_teacher_reference"] = assess(
                "qat-teacher-reference",
                reference=True,
                candidate=root / inputs["teacher"],
            )
        behavior = {"checkpoint_sha256": inputs["checkpoint_sha256"], "reports": []}
        for style in MIXTURE_STYLES:
            path = assess("opponent-" + style, extra=("--opponent", style))
            behavior["reports"].append({"path": path, "role": "scripted"})
        for index, path in enumerate(inputs["historical"]):
            result = assess(
                f"historical-{index}", extra=("--opponent-checkpoint", str(root / path))
            )
            behavior["reports"].append(
                {"path": result, "role": "historical", "checkpoint": path}
            )
        for index, path in enumerate(inputs["partners"]):
            result = assess(
                f"crossplay-{index}",
                players=4,
                extra=("--partner-checkpoint", str(root / path)),
            )
            behavior["reports"].append(
                {"path": result, "role": "crossplay", "checkpoint": path}
            )
        for style in ("literal", "cautious", "call-heavy", "silent", "ignore"):
            result = assess("partner-" + style, players=4, extra=("--partner", style))
            behavior["reports"].append({"path": result, "role": "human-partner"})
        atomic_json(root / "behavior.json", behavior)
        evidence["behavior"] = "behavior.json"
        reference = assess("population-reference", extra=("--opponent", "population"))
        ablated = assess(
            "population-no-adaptation",
            extra=(
                "--opponent",
                "population",
                "--ablation",
                "adaptation",
                "--compare-reference",
                str(root / reference),
            ),
        )
        atomic_json(
            root / "adaptation-evidence.json",
            {"reference": reference, "ablated": ablated},
        )
        evidence["adaptation_evidence"] = "adaptation-evidence.json"
        for partner in ("learned", "literal"):
            reference = assess(
                "communication-" + partner,
                players=4,
                extra=("--opponent", "population", "--partner", partner),
            )
            ablated = assess(
                "no-communication-" + partner,
                players=4,
                extra=(
                    "--opponent",
                    "population",
                    "--partner",
                    partner,
                    "--ablation",
                    "signals",
                    "--compare-reference",
                    str(root / reference),
                ),
            )
            atomic_json(
                root / f"signals-{partner}-evidence.json",
                {"reference": reference, "ablated": ablated},
            )
        evidence["signals_evidence"] = "signals-literal-evidence.json"
        for style in ("baseline", "cautious"):
            output = root / "reports" / f"bluff-{style}.json"
            run(
                "bluff-" + style,
                [
                    "retruco_ml.bluff",
                    str(checkpoint),
                    "--quantized",
                    str(bundle / "policy.bin"),
                    "--pairs",
                    str(args.bluff_pairs),
                    "--seed",
                    str(reservation["bluff_seed"]),
                    "--opponent",
                    style,
                    "--output",
                    str(output),
                ],
                output,
            )
        evidence["bluff"] = (
            "reports/bluff-baseline.json"  # Preregistered primary; report cautious too, without cherry-picking.
        )
        exploiters = {"checkpoint_sha256": inputs["checkpoint_sha256"], "runs": []}
        if args.mode != "validation":
            for players in (2, 4):
                for seed in (901, 902, 903):
                    folder = root / "exploiters" / f"{players}-seats-{seed}"
                    folder.mkdir(parents=True, exist_ok=True)
                    path = folder / "checkpoint.pt"
                    saved = (
                        torch.load(path, map_location="cpu", weights_only=True)
                        if path.exists()
                        else None
                    )
                    remaining = max(
                        0,
                        args.exploiter_hours
                        - (saved.get("total_seconds", 0) / 3600 if saved else 0),
                    )
                    if (not saved) or (args.mode == "release" and remaining > 0):
                        command = [
                            "retruco_ml.train",
                            "--output",
                            str(folder),
                            "--seed",
                            str(seed),
                            "--players",
                            str(players),
                            "--device",
                            "cpu",
                            "--opponent-checkpoint",
                            str(checkpoint),
                            "--opponent-quantized",
                            str(bundle / "policy.bin"),
                            "--hours",
                            str(remaining or args.exploiter_hours),
                            "--workers",
                            str(args.workers),
                            "--environments",
                            str(args.environments),
                        ]
                        if saved:
                            command.append("--resume")
                        if args.mode == "smoke":
                            command.extend(["--updates", "1", "--steps", "64"])
                        run(f"train-exploiter-{players}-{seed}", command, path)
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if (
                        saved.get("opponent_checkpoint_sha256")
                        != inputs["checkpoint_sha256"]
                        or saved.get("opponent_model_sha256") != model_hash
                        or saved.get("initialization")
                        != {"kind": "random", "seed": seed}
                    ):
                        raise ValueError("exploiter training identity mismatch")
                    assessment = assess(
                        f"exploiter-{players}-{seed}",
                        players=players,
                        extra=("--opponent-checkpoint", str(path)),
                    )
                    exploiters["runs"].append(
                        {
                            "players": players,
                            "seed": seed,
                            "initialization": "fresh",
                            "training_seconds": saved["total_seconds"],
                            "target_model_sha256": model_hash,
                            "checkpoint": str(path.relative_to(root)),
                            "checkpoint_sha256": digest(path),
                            "assessment": assessment,
                        }
                    )
            atomic_json(root / "exploiters.json", exploiters)
            evidence["exploiters"] = "exploiters.json"
        if args.verification:
            for key, file in (
                ("integration", "integration.json"),
                ("performance", "performance.json"),
                ("runtime_parity", "parity.json"),
            ):
                evidence[key] = str(args.verification / file)
        atomic_json(
            root / "campaign.json",
            {
                **configuration,
                "fingerprint": fingerprint,
                "seed_reservation": reservation,
                "checkpoint_sha256": inputs["checkpoint_sha256"],
                "model_sha256": model_hash,
                "evaluation_source_commit": snapshot["source_commit"],
                "source_dirty": snapshot["source_dirty"],
                "source_manifest": "source/manifest.json",
                "input_manifest": "inputs/manifest.json",
            },
        )
        evidence["evaluation_campaign"] = "campaign.json"
        atomic_json(root / "evidence.json", evidence)
        # Audit is intentionally a report, not an approval mutation. In smoke or
        # validation mode its holdout/sample gates must fail. Human review is
        # supplied later, after the selected model is ready for actual play.
        from .release_audit import Audit

        audit = Audit(root / "evidence.json").run()
        atomic_json(root / "audit.json", audit)
        status.update(
            status="complete",
            stage="evidence-collected",
            child_pid=None,
            unmet=[c["name"] for c in audit["checks"] if not c["passed"]],
        )
    except BaseException as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        atomic_json(root / "status.json", status)


if __name__ == "__main__":
    main()
