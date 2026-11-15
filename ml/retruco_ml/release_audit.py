"""Read-only release audit. Recompute primary metrics; never trust green flags.

Evidence paths are relative to the supplied evidence JSON. Missing evidence is
an unmet gate, not a waived requirement. This command does not approve or deploy.
"""

import argparse
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch

from .artifacts import digest, read_quantized, validate_manifest
from .assessment import ASSESSMENT_VERSION, SEAT_SCHEDULE, load_checkpoint
from .bundle import training_commit
from .evaluation_styles import MIXTURE_STYLES, POPULATION_SCHEDULE, UNSEEN_STYLES
from .export import serialize
from .holdout import validate_reservations
from .model_capacity import MAX_ACTOR_PARAMETERS
from .parity import differences
from .splits import BLUFF_END, BLUFF_START, RELEASE_END, RELEASE_START
from .statistics import interval, paired_loss, pooled
from .strength_targets import MODE_WIN_TARGETS

CONFIGURATIONS = {
    (players, target, flor)
    for players in (2, 4)
    for target in (12, 24)
    for flor in (False, True)
}
REQUIRED_INTEGRATION = {
    "native-tests",
    "wasm-tests",
    "frontend-tests",
    "typecheck",
    "browser-1v1",
    "browser-2v2",
    "controller-handoff",
    "server-restore",
    "corrupt-artifacts",
    "stale-work",
    "soak",
    "production-approval-guard",
}


def strict_json(data):
    def reject(value):
        raise ValueError("nonfinite JSON value: " + value)

    def finite(value):
        result = float(value)
        if not math.isfinite(result):
            reject(value)
        return result

    return json.loads(data, parse_constant=reject, parse_float=finite)


def browser_performance(report, model_hash, wasm_hash):
    if report["modelSha256"] != model_hash or report["wasmSha256"] != wasm_hash:
        raise ValueError("benchmark belongs to different artifacts")
    if (
        report.get("measurementVersion") != 2
        or report.get("decisionPath") != "actual-worker-and-decision-session"
    ):
        raise ValueError("benchmark must exercise the complete worker decision path")
    sources = report.get("sourceSha256", {})
    base_sources = {"model", "fallback", "decision", "load-policy", "worker"}
    recurrent = bool(report.get("recurrentScope"))
    if (
        set(sources) not in (base_sources, base_sources | {"memory"})
        or (recurrent and "memory" not in sources)
        or any(not re.fullmatch("[0-9a-f]{64}", v) for v in sources.values())
    ):
        raise ValueError("complete worker source identities absent")
    profiles = {p["rate"]: p for p in report["profiles"]}
    if set(profiles) != {1, 6, 10} or len(report["profiles"]) != 3:
        raise ValueError("missing or duplicate CPU profiles")
    for rate, profile in profiles.items():
        worker = (
            profile.get("workerThrottleVerified")
            and not profile.get("workerThrottleError")
            and profile["workerCalibrationRatio"] >= rate * 0.55
        )
        if not worker and not (
            profile.get("pageThrottleVerified")
            and profile["pageCalibrationRatio"] >= rate * 0.55
        ):
            raise ValueError("throttling was not independently calibrated")
        if not any(c["actions"] == 1024 for c in profile["cases"]):
            raise ValueError("capacity case absent")
        if not any(
            c.get("history", {}).get("signals") == 20
            and c["history"].get("cantos", 0) >= 16
            for c in profile["cases"]
        ):
            raise ValueError("full-history case absent")
        for case in profile["cases"]:
            values = case["roundTrip" if worker else "estimatedComplete"]
            if (
                values["samples"] < 200
                or not 0 <= values["p95Ms"] <= 20
                or not values["p95Ms"] <= values["p99Ms"] <= 50
            ):
                raise ValueError("complete decision latency gate failed")
        if not 0 < profile["startup"]["worker"]["memoryBytes"] <= 4 * 1024 * 1024:
            raise ValueError("WASM memory gate failed")
        if recurrent:
            if not any(c.get("events") == 2048 for c in profile["cases"]):
                raise ValueError("maximum recurrent-history case absent")
            if any(
                c["reconnect"]["roundTripMs" if worker else "estimatedCompleteMs"]
                > 1000
                for c in profile["cases"]
            ):
                raise ValueError("recurrent replay budget exceeded")
            memory = profile["memory"]
            if (
                sum(
                    memory.get(k, 0)
                    for k in ("usedSize", "backingStorageSize", "embedderHeapUsedSize")
                )
                > 64 * 1024 * 1024
            ):
                raise ValueError("recurrent worker memory budget exceeded")
        cold = profile["cold"]
        if (
            not cold.get("pageThrottleVerified")
            or cold["pageCalibrationRatio"] < rate * 0.55
        ):
            raise ValueError("cold initialization throttling not calibrated")
        for target in ("page", "worker"):
            if (
                not 0 <= cold[target]["readyMs"] <= cold[target]["totalMs"] <= 500
                or not 0 < cold[target]["memoryBytes"] <= 4 * 1024 * 1024
            ):
                raise ValueError("cold initialization budget exceeded")
    if not 0 < report["assets"]["totalGzipBytes"] <= 256 * 1024:
        raise ValueError("transfer budget exceeded")
    if (
        report["productionGuard"].get("productionEvaluationDisabled") is not True
        or report["productionGuard"].get("unapprovedBinaryDownloads") != 0
    ):
        raise ValueError("production approval guard evidence absent")
    return {"scope": report["decisionPath"], "assets": report["assets"]}


def integration_evidence(report, root, model_hash, backend_commit):
    if report.get("complete") is not True or report.get("smoke") is not False:
        raise ValueError("integration verification is incomplete or only a smoke run")
    verified = set()
    for row in report["checks"]:
        path = root / row["evidence_path"]
        # A successful Go build or input generator can legitimately be silent.
        # Mandatory test evidence must still contain output, and every receipt
        # must bind its command, exit status and exact log bytes.
        if (
            not row.get("command")
            or not path.is_file()
            or digest(path) != row["evidence_sha256"]
        ):
            raise ValueError("missing or changed integration evidence")
        if row["name"] in REQUIRED_INTEGRATION and not path.stat().st_size:
            raise ValueError("empty mandatory integration test evidence")
        if row["exit_code"] != 0:
            raise ValueError("failed integration command: " + row["name"])
        verified.add(row["name"])
    if not REQUIRED_INTEGRATION.issubset(verified):
        raise ValueError("integration validation coverage incomplete")
    if report["model_sha256"] != model_hash:
        raise ValueError("integration model mismatch")
    if (
        report.get("source_dirty") is not False
        or report["backend_commit"] != backend_commit
    ):
        raise ValueError("integration backend source differs from candidate")
    if not re.fullmatch("[0-9a-f]{40}", report.get("frontend_commit") or ""):
        raise ValueError("missing integration frontend revision")
    return sorted(verified)


def configurations(report, expected=CONFIGURATIONS, minimum=2000):
    if (
        report.get("assessment_version") != ASSESSMENT_VERSION
        or report.get("seat_schedule") != SEAT_SCHEDULE
    ):
        raise ValueError("assessment schema or seat schedule mismatch")
    if report.get("opponent") == "population" and (
        report.get("opponent_schedule") != POPULATION_SCHEDULE
        or report.get("opponent_styles") != list(MIXTURE_STYLES)
    ):
        raise ValueError("unknown evaluation opponent population")
    rows = report.get("configurations", [])
    keys = [(r["players"], r["target"], r["flor"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("missing, duplicate or unexpected rules configurations")
    counts = set()
    for row in rows:
        count = row["pairs"]
        values = row["pair_scores"]
        if (
            type(row["flor"]) is not bool
            or type(count) is not int
            or count < minimum
            or len(values) != count
        ):
            raise ValueError("insufficient or inconsistent paired-deal count")
        if any(type(x) not in (int, float) or x not in (0, 0.5, 1) for x in values):
            raise ValueError("pair scores must represent two actual wins/losses")
        counts.add(count)
    if len(counts) != 1:
        raise ValueError("seed clusters do not align across configurations")
    seed = report["seed"]
    if (
        type(seed) is not int
        or seed < RELEASE_START
        or seed + max(counts) > RELEASE_END
    ):
        raise ValueError("release assessment used a non-holdout seed range")
    return rows


def quality(report, checkpoint_hash, model_hash):
    if (
        report["checkpoint_sha256"] != checkpoint_hash
        or report["quantized_sha256"] != model_hash
    ):
        raise ValueError("assessment is for a different checkpoint or model")
    if (report["opponent"], report["partner"], report["ablation"]) != (
        "baseline",
        "learned",
        "none",
    ):
        raise ValueError(
            "primary strength assessment must use the unchanged baseline matchup"
        )
    rows = configurations(report)
    summary = pooled(rows)
    modes = {}
    for players in (2, 4):
        mode = pooled([r for r in rows if r["players"] == players])
        modes[str(players)] = mode
        target = MODE_WIN_TARGETS[players]
        if mode["win_rate"] < target:
            raise ValueError(
                f"{players}-player win rate is below required {target:.0%}"
            )
        mode["target_win_rate"] = target
        if mode["cluster_ci95"][0] <= 0.5:
            raise ValueError(f"{players}-player improvement is not established")
    for row in rows:
        if interval(row["pair_scores"])[1] < 0.5:
            raise ValueError("a rules configuration shows a clear regression")
    return {"pooled": summary, "modes": modes}


def quantization(reference, quantized, checkpoint_hash, model_hash):
    configurations(reference)
    configurations(quantized)
    if (
        reference["quantized_sha256"] is not None
        or quantized["quantized_sha256"] != model_hash
    ):
        raise ValueError(
            "expected float32 reference and exact deployed quantized model"
        )
    for key in ("checkpoint_sha256", "opponent", "partner", "ablation", "seed"):
        if reference[key] != quantized[key]:
            raise ValueError("unpaired quantization comparison")
    if reference["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("checkpoint mismatch")
    loss = paired_loss(reference, quantized)
    if loss["cluster_ci95"][1] > 0.01:
        raise ValueError("quantization loss upper bound exceeds one percentage point")
    return loss


def teacher_preservation(
    reference, quantized, teacher_hash, checkpoint_hash, model_hash
):
    configurations(reference)
    configurations(quantized)
    if (
        reference["checkpoint_sha256"] != teacher_hash
        or reference["quantized_sha256"] is not None
    ):
        raise ValueError("QAT reference is not the original float32 teacher")
    if (
        quantized["checkpoint_sha256"] != checkpoint_hash
        or quantized["quantized_sha256"] != model_hash
    ):
        raise ValueError("QAT comparison is not the deployed candidate")
    for key in ("opponent", "partner", "ablation", "seed"):
        if reference[key] != quantized[key]:
            raise ValueError("unpaired QAT teacher comparison")
    if (reference["opponent"], reference["partner"], reference["ablation"]) != (
        "baseline",
        "learned",
        "none",
    ):
        raise ValueError("QAT teacher comparison must retain the primary matchup")
    loss = paired_loss(reference, quantized)
    if loss["cluster_ci95"][1] > 0.01:
        raise ValueError(
            "QAT loss against the original teacher exceeds one percentage point"
        )
    return loss


class Audit:
    def __init__(self, path):
        self.root = path.resolve().parent
        self.evidence = strict_json(path.read_text())
        self.checks = []
        self.hashes = {}
        self.evaluation_fingerprint = None
        self.evaluation_source_commit = None
        self.evaluation_base = None

    def path(self, name):
        value = self.evidence.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError("missing evidence: " + name)
        return (self.root / value).resolve()

    def read(self, name):
        path = self.path(name)
        data = path.read_bytes()
        self.hashes[name] = hashlib.sha256(data).hexdigest()
        value = strict_json(data)
        self.assessment_identity(value, path)
        return value

    def read_report(self, path):
        data = path.read_bytes()
        value = strict_json(data)
        self.assessment_identity(value, path)
        self.hashes["report:" + str(path.relative_to(self.root))] = hashlib.sha256(
            data
        ).hexdigest()
        return value

    def assessment_identity(self, report, path=None):
        if self.evaluation_fingerprint and (
            "configurations" in report or "fixture" in report
        ):
            if (
                report.get("evaluation_fingerprint") != self.evaluation_fingerprint
                or report.get("evaluation_source_commit")
                != self.evaluation_source_commit
            ):
                raise ValueError(
                    "assessment belongs to a different frozen evaluation campaign"
                )
            if path is not None:
                if path.resolve().parent != self.evaluation_base / "reports":
                    raise ValueError("assessment is outside the frozen campaign")
                receipt = strict_json(
                    (self.evaluation_base / "receipts" / path.name).read_text()
                )
                log = self.evaluation_base / "logs" / (path.stem + ".log")
                if (
                    receipt["exit_code"] != 0
                    or receipt["output_sha256"] != digest(path)
                    or receipt["log_sha256"] != digest(log)
                ):
                    raise ValueError("assessment receipt or raw output changed")

    def check(self, name, callback):
        try:
            detail = callback()
            self.checks.append({"name": name, "passed": True, "detail": detail})
        except (ValueError, KeyError, TypeError, OSError, AssertionError) as error:
            self.checks.append({"name": name, "passed": False, "reason": str(error)})

    def run(self):
        try:
            folder = self.path("bundle")
            manifest_data = (folder / "manifest.json").read_bytes()
            manifest = strict_json(manifest_data)
            self.hashes["bundle_manifest"] = hashlib.sha256(manifest_data).hexdigest()
            model = load_checkpoint(self.path("checkpoint"))
            model_hash = manifest["sha256"]
            checkpoint_hash = model.checkpoint_sha256
        except (ValueError, KeyError, TypeError, OSError) as error:
            return {
                "ready": False,
                "checks": [
                    {"name": "candidate", "passed": False, "reason": str(error)}
                ],
            }

        def identity(report):
            if report.get("checkpoint_sha256") != checkpoint_hash:
                raise ValueError("candidate checkpoint mismatch")
            self.assessment_identity(report)

        def artifacts():
            data, reference = serialize(model)
            validate_manifest(
                manifest,
                (folder / "policy.bin").read_bytes(),
                (folder / "policy.wasm").read_bytes(),
            )
            if (folder / "policy.bin").read_bytes() != data or (
                folder / "reference.f32"
            ).read_bytes() != reference:
                raise ValueError("bundle weights are not the export of the checkpoint")
            if (
                hashlib.sha256(data).hexdigest() != model_hash
                or manifest["checkpoint_sha256"] != checkpoint_hash
            ):
                raise ValueError("bundle checkpoint/model hash mismatch")
            if digest(folder / "policy.wasm") != manifest["wasm_sha256"]:
                raise ValueError("evaluator hash mismatch")
            if manifest.get("source_dirty") is not False:
                raise ValueError("bundle was built from dirty source")
            for key in (
                "exporter_commit",
                "runtime_source_commit",
                "training_source_commit",
            ):
                if not re.fullmatch("[0-9a-f]{40}", manifest.get(key) or ""):
                    raise ValueError("missing provenance: " + key)
            if (
                digest(folder / "training-source-manifest.json")
                != manifest["training_snapshot_sha256"]
            ):
                raise ValueError("training snapshot provenance mismatch")
            if manifest.get("training_kind", "ppo") != model.training_kind:
                raise ValueError("training kind mismatch")
            if model.training_kind == "qat":
                for key, value in model.qat_origin.items():
                    if not value or manifest.get(key) != value:
                        raise ValueError("QAT teacher provenance mismatch")
                if not re.fullmatch(
                    "[0-9a-f]{40}", manifest["teacher_training_source_commit"]
                ):
                    raise ValueError("QAT teacher source commit missing")
                if (
                    digest(folder / "teacher-training-source-manifest.json")
                    != manifest["teacher_training_snapshot_sha256"]
                ):
                    raise ValueError("QAT teacher source manifest changed")
            if (
                manifest["actor_parameters"] > MAX_ACTOR_PARAMETERS
                or len(data) > 128 * 1024
            ):
                raise ValueError("model exceeds footprint budget")
            return {"model_sha256": model_hash, "checkpoint_sha256": checkpoint_hash}

        self.check("artifact_identity_and_provenance", artifacts)

        def evaluation_provenance():
            campaign = self.read("evaluation_campaign")
            identity(campaign)
            base = self.path("evaluation_campaign").parent
            if (
                campaign["model_sha256"] != model_hash
                or campaign["mode"] != "release"
                or campaign["source_dirty"] is not False
            ):
                raise ValueError(
                    "release assessment must use a clean, frozen release campaign"
                )
            configuration = strict_json((base / "configuration.json").read_text())
            fingerprint = hashlib.sha256(
                json.dumps(
                    configuration, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            if fingerprint != campaign["fingerprint"]:
                raise ValueError("evaluation configuration changed")
            source = base / campaign["source_manifest"]
            inputs = base / campaign["input_manifest"]
            if (
                digest(source) != configuration["source_manifest_sha256"]
                or digest(inputs) != configuration["inputs_sha256"]
            ):
                raise ValueError("evaluation source/input provenance changed")
            repository = Path(
                os.environ.get(
                    "RETRUCO_ML_REPOSITORY", Path(__file__).resolve().parents[2]
                )
            )
            revision = training_commit(source, repository)
            if revision != campaign["evaluation_source_commit"]:
                raise ValueError("evaluation source revision mismatch")
            frozen = strict_json(inputs.read_text())
            if (
                frozen["model_sha256"] != model_hash
                or frozen["checkpoint_sha256"] != checkpoint_hash
            ):
                raise ValueError("evaluation inputs belong to another candidate")
            for name, expected in frozen["files"].items():
                if digest(base / name) != expected:
                    raise ValueError("frozen evaluation input changed")
            reservation = strict_json((base / "seed-reservation.json").read_text())
            if (
                reservation != campaign["seed_reservation"]
                or reservation["fingerprint"] != fingerprint
            ):
                raise ValueError("test seed reservation changed")
            ledger = strict_json(
                (repository / "ml/runs/release-seeds.json").read_text()
            )
            validate_reservations(ledger["reservations"])
            if ledger["reservations"].get(fingerprint) != reservation:
                raise ValueError("test range was not reserved in the campaign ledger")
            self.evaluation_fingerprint = fingerprint
            self.evaluation_source_commit = revision
            self.evaluation_base = base
            for name in ("reference", "quantized"):
                result = self.read(name)
                if result["seed"] != reservation["seed"] or any(
                    c["pairs"] != reservation["pairs"] for c in result["configurations"]
                ):
                    raise ValueError("primary reports used different reserved deals")
            return {
                "source_commit": revision,
                "fingerprint": fingerprint,
                "seed_reservation": reservation,
            }

        self.check("evaluation_provenance_and_holdout", evaluation_provenance)
        self.check(
            "deployed_playing_strength",
            lambda: quality(self.read("quantized"), checkpoint_hash, model_hash),
        )
        self.check(
            "quantization_loss",
            lambda: quantization(
                self.read("reference"),
                self.read("quantized"),
                checkpoint_hash,
                model_hash,
            ),
        )
        if model.training_kind == "qat":
            self.check(
                "qat_teacher_strength_preservation",
                lambda: teacher_preservation(
                    self.read("qat_teacher_reference"),
                    self.read("quantized"),
                    model.qat_origin["teacher_checkpoint_sha256"],
                    checkpoint_hash,
                    model_hash,
                ),
            )

        def performance():
            return browser_performance(
                self.read("performance"), model_hash, manifest["wasm_sha256"]
            )

        self.check("browser_performance", performance)

        def parity():
            report = self.read("runtime_parity")
            identity(report)
            if (
                report["model_sha256"] != model_hash
                or report["wasm_sha256"] != manifest["wasm_sha256"]
            ):
                raise ValueError("parity artifact mismatch")
            if report["cases"] < 20 or report["max_candidates"] != 1024:
                raise ValueError("insufficient parity cases")
            if set(report["engines"]) != {
                "pytorch",
                "native-rust",
                "node-wasm",
                "browser-wasm",
            }:
                raise ValueError("not all inference engines verified")
            evidence_path = self.path("runtime_parity").parent / report["evidence_path"]
            if digest(evidence_path) != report["evidence_sha256"]:
                raise ValueError("parity raw evidence changed")
            evidence = strict_json(evidence_path.read_text())
            cases = evidence["cases"]
            if (
                len(cases) != report["cases"]
                or sum(not c["synthetic"] for c in cases) < 20
            ):
                raise ValueError("insufficient authorized parity observations")
            if max(len(c["actions"]) for c in cases) != 1024 or not any(
                c["signal"] for c in cases
            ):
                raise ValueError("capacity or signal parity evidence absent")
            for name in ("float32", "quantized", "native", "node", "browser"):
                if len(evidence[name]) != len(cases):
                    raise ValueError("parity output count mismatch")
                for case, values in zip(cases, evidence[name], strict=True):
                    if len(values) != (6 if case["signal"] else len(case["actions"])):
                        raise ValueError("parity output shape mismatch")
            measured = {
                name: differences(evidence["quantized"], evidence[name])
                for name in ("native", "node", "browser")
            }
            if max(row["max_probability_error"] for row in measured.values()) > 1e-4:
                raise ValueError("numerical parity failed")
            return {
                "runtime_errors": measured,
                "quantization": differences(evidence["float32"], evidence["quantized"]),
                "evidence_sha256": report["evidence_sha256"],
            }

        self.check("candidate_runtime_parity", parity)

        def behavior():
            evidence = self.read("behavior")
            identity(evidence)
            reports = evidence["reports"]
            coverage = {
                "scripted": set(),
                "historical": set(),
                "crossplay": set(),
                "human-partner": set(),
            }
            candidate_data = self.path("checkpoint").read_bytes()
            if hashlib.sha256(candidate_data).hexdigest() != checkpoint_hash:
                raise ValueError("candidate checkpoint changed during audit")
            saved_candidate = torch.load(
                io.BytesIO(candidate_data), map_location="cpu", weights_only=True
            )
            candidate_seed = saved_candidate.get(
                "initial_seed", saved_candidate.get("config", {}).get("seed")
            )
            partner_seeds = set()
            actor_hashes = {"historical": set(), "crossplay": set()}
            for item in reports:
                report = self.read_report(self.root / item["path"])
                identity(report)
                is_partner = item["role"] in ("crossplay", "human-partner")
                expected = (
                    {c for c in CONFIGURATIONS if c[0] == 4}
                    if is_partner
                    else CONFIGURATIONS
                )
                configurations(report, expected)
                if report["quantized_sha256"] != model_hash:
                    raise ValueError("robustness evaluated a different precision/model")
                if report["ablation"] != "none":
                    raise ValueError(
                        "robustness assessment unexpectedly ablated the policy"
                    )
                policy = report["partner"] if is_partner else report["opponent"]
                if item["role"] in ("crossplay", "historical"):
                    path = self.root / item["checkpoint"]
                    if digest(path) != policy or policy == checkpoint_hash:
                        raise ValueError(
                            "opponent/partner checkpoint identity mismatch"
                        )
                    other = load_checkpoint(path)
                    if other.checkpoint_sha256 != policy:
                        raise ValueError(
                            "opponent/partner checkpoint changed during audit"
                        )
                    actor_hash = hashlib.sha256(serialize(other)[0]).hexdigest()
                    if actor_hash == model_hash:
                        raise ValueError("coverage repeats the candidate actor weights")
                    actor_hashes[item["role"]].add(actor_hash)
                    if item["role"] == "crossplay":
                        partner_data = path.read_bytes()
                        if hashlib.sha256(partner_data).hexdigest() != policy:
                            raise ValueError("partner checkpoint changed during audit")
                        saved_partner = torch.load(
                            io.BytesIO(partner_data),
                            map_location="cpu",
                            weights_only=True,
                        )
                        partner_seed = saved_partner.get(
                            "initial_seed", saved_partner.get("config", {}).get("seed")
                        )
                        if (
                            type(candidate_seed) is not int
                            or type(partner_seed) is not int
                            or partner_seed == candidate_seed
                        ):
                            raise ValueError(
                                "cross-play partner does not have an independent training seed"
                            )
                        partner_seeds.add(partner_seed)
                coverage[item["role"]].add(policy)
            if not {
                "cautious",
                "aggressive",
                "call-heavy",
                "literal",
                "deceptive",
                "silent",
            }.issubset(coverage["scripted"]):
                raise ValueError("scripted opponent coverage incomplete")
            if not set(UNSEEN_STYLES).issubset(coverage["scripted"]):
                raise ValueError("held-out scripted opponent coverage incomplete")
            if len(coverage["historical"]) < 2 or len(coverage["crossplay"]) < 2:
                raise ValueError("independent/historical policy coverage incomplete")
            if len(partner_seeds) < 2 or any(
                len(values) < 2 for values in actor_hashes.values()
            ):
                raise ValueError(
                    "coverage repeats actor weights or independent partner seeds"
                )
            if not {"literal", "ignore", "silent"}.issubset(coverage["human-partner"]):
                raise ValueError("human-like partner coverage incomplete")
            return {k: sorted(v) for k, v in coverage.items()}

        self.check("opponent_and_partner_coverage", behavior)
        for mode in ("adaptation", "signals"):

            def ablation(mode=mode):
                pair = self.read(mode + "_evidence")
                reference = self.read_report(self.root / pair["reference"])
                changed = self.read_report(self.root / pair["ablated"])
                identity(reference)
                identity(changed)
                if reference["ablation"] != "none" or changed["ablation"] != mode:
                    raise ValueError("ablation mode mismatch")
                for key in (
                    "opponent",
                    "partner",
                    "quantized_sha256",
                    "opponent_schedule",
                    "opponent_styles",
                ):
                    if reference.get(key) != changed.get(key):
                        raise ValueError("ablation changes another factor")
                expected = (
                    CONFIGURATIONS
                    if mode == "adaptation"
                    else {c for c in CONFIGURATIONS if c[0] == 4}
                )
                configurations(reference, expected)
                configurations(changed, expected)
                if reference["quantized_sha256"] != model_hash:
                    raise ValueError("ablation is not deployed policy")
                loss = paired_loss(reference, changed)
                if loss["cluster_ci95"][0] <= 0:
                    raise ValueError("benefit is not statistically established")
                return loss

            self.check(mode + "_benefit", ablation)

        def bluff():
            report = self.read("bluff")
            identity(report)
            if report["quantized_sha256"] != model_hash:
                raise ValueError("bluff diagnostic is not the deployed policy")
            if (
                report["pairs"] < 1000
                or report["seed"] < BLUFF_START
                or report["seed"] + report["pairs"] > BLUFF_END
            ):
                raise ValueError("insufficient held-out bluff sampling")
            gains = report["paired_win_gains"]
            if len(gains) != report["pairs"] or any(x not in (-1, 0, 1) for x in gains):
                raise ValueError("invalid counterfactual pairs")
            if (
                not 0 < report["policy_raise_probability"] <= 1
                or interval(gains)[0] <= 0
            ):
                raise ValueError(
                    "profitable played bluff not established at match level"
                )
            return {"win_gain": float(np.mean(gains)), "ci95": interval(gains)}

        self.check("conditional_bluff_value", bluff)

        def exploiters():
            report = self.read("exploiters")
            identity(report)
            seeds = {2: set(), 4: set()}
            target = read_quantized(
                (folder / "policy.bin").read_bytes(), model
            ).state_dict()
            for row in report["runs"]:
                if row["training_seconds"] < 3600 or row["initialization"] != "fresh":
                    raise ValueError("insufficient fresh best-response training")
                checkpoint_data = (self.root / row["checkpoint"]).read_bytes()
                exploiter_hash = hashlib.sha256(checkpoint_data).hexdigest()
                if exploiter_hash != row["checkpoint_sha256"]:
                    raise ValueError("exploiter checkpoint changed")
                saved = torch.load(
                    io.BytesIO(checkpoint_data), map_location="cpu", weights_only=True
                )
                if (
                    saved["total_seconds"] < 3600
                    or saved["initial_seed"] != row["seed"]
                    or saved["config"]["players"] != row["players"]
                ):
                    raise ValueError("exploiter training metadata mismatch")
                if saved.get("initialization") != {
                    "kind": "random",
                    "seed": row["seed"],
                }:
                    raise ValueError(
                        "fresh exploiter initialization is not established"
                    )
                if (
                    saved["opponent_checkpoint_sha256"] != checkpoint_hash
                    or saved["opponent_model_sha256"] != model_hash
                ):
                    raise ValueError("exploiter trained against a different target")
                if set(saved["fixed_opponent"]) != set(target) or any(
                    not torch.equal(saved["fixed_opponent"][name], value)
                    for name, value in target.items()
                ):
                    raise ValueError(
                        "saved exploiter target weights differ from candidate"
                    )
                result = self.read_report(self.root / row["assessment"])
                identity(result)
                configurations(
                    result, {c for c in CONFIGURATIONS if c[0] == row["players"]}
                )
                if (
                    result["opponent"] != exploiter_hash
                    or result["quantized_sha256"] != model_hash
                    or result["ablation"] != "none"
                ):
                    raise ValueError(
                        "vulnerability evaluation matchup differs from declared target"
                    )
                if row["target_model_sha256"] != model_hash:
                    raise ValueError("exploiter targeted another model")
                seeds[row["players"]].add(row["seed"])
            if any(len(values) < 3 for values in seeds.values()):
                raise ValueError("need three independent exploiters per mode")
            return {str(k): sorted(v) for k, v in seeds.items()}

        self.check("empirical_vulnerability_assessment", exploiters)

        def integration():
            report = self.read("integration")
            identity(report)
            return integration_evidence(
                report,
                self.path("integration").parent,
                model_hash,
                manifest["exporter_commit"],
            )

        self.check("integration_and_recovery", integration)

        def human():
            review = self.read("human_review")
            identity(review)
            if review["model_sha256"] != model_hash or set(review["played_modes"]) != {
                2,
                4,
            }:
                raise ValueError(
                    "human review does not cover this model and both modes"
                )
            if (
                not review.get("reviewer")
                or not review.get("notes")
                or review.get("coordination_acceptable") is not True
                or review.get("weaknesses_reviewed") is not True
            ):
                raise ValueError("human coordination/weakness review incomplete")
            return {
                "reviewer": review["reviewer"],
                "played_modes": review["played_modes"],
            }

        self.check("human_partnership_review", human)
        return {
            "ready": all(c["passed"] for c in self.checks),
            "model_sha256": model_hash,
            "checkpoint_sha256": checkpoint_hash,
            "source_commits": {
                "training": manifest.get("training_source_commit"),
                "exporter": manifest.get("exporter_commit"),
                "runtime": manifest.get("runtime_source_commit"),
                "evaluation": self.evaluation_source_commit,
                "teacher": manifest.get("teacher_training_source_commit"),
            },
            "checks": self.checks,
            "evidence_hashes": self.hashes,
            "approval_changed": False,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = Audit(args.evidence).run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "ready": report["ready"],
                "unmet": [c["name"] for c in report["checks"] if not c["passed"]],
            }
        )
    )
    if not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
