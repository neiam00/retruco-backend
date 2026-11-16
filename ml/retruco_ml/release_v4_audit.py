"""Independently recompute the full format-4 release verdict. Does not change approval."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .artifacts import digest
from .assessment import load_checkpoint
from .bundle import training_commit
from .controlled_train import atomic_json
from .export import serialize
from .holdout import validate_reservations
from .parity import differences
from .release_audit import (
    CONFIGURATIONS,
    browser_performance,
    configurations,
    integration_evidence,
    quality,
    quantization,
)
from .release_v4_campaign import (
    attest,
    check_inputs,
    fingerprint,
    load,
    response_checkpoint,
)
from .selective_bluff import MATCHES as OPENING_MATCHES
from .selective_bluff import PROTOCOL as OPENING_PROTOCOL
from .selective_bluff import start_seed, verify_ordinary
from .statistics import interval, paired_loss


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def native_human_primary(base, project, conf):
    configuration = load(base / "paired-configuration.json")
    row = load(base / "seed-reservation.json")
    ident = fingerprint(configuration)
    ledger = load(project / "ml/runs/release-seeds.json")["reservations"]
    validate_reservations(ledger)
    require(
        ledger.get(ident) == row and row["pairs"] == 2000,
        "Unregistered paired human benchmark",
    )
    ids = configuration["hashes"]
    require(
        (ids["model"], ids["wasm"], ids["checkpoint"], ids["simulator"])
        == (
            conf["model_sha256"],
            conf["wasm_sha256"],
            conf["checkpoint_sha256"],
            conf["files"]["bin/selfplay"],
        ),
        "Primary identities differ",
    )
    require(
        digest(base / "paired-configuration.json") == conf["human_primary_plan_sha256"],
        "Changed preregistered human benchmark",
    )
    require(
        digest(base / "paired_release.py") == ids["release_driver"],
        "Changed primary evaluation driver",
    )
    s = load(base / "paired-results.json")
    expected = {
        "native-headsup": (2, "reference-cpu"),
        "native-literal-v2": (4, "literal-v2"),
        "native-trusting-v2": (4, "trusting-v2"),
        "native-cautious-v2": (4, "cautious-v2"),
        "native-reference-human": (4, "reference-human"),
    }
    per_precision = {}
    modes = {}
    losses = []
    partners = {}
    for v in ("int8", "fp32"):
        for k, (n, partner) in expected.items():
            p = base / f"{v}-{k}.json"
            d = load(p)
            require(
                digest(p) == s["reports"][p.stem]
                and d["release_fingerprint"] == ident
                and d["seed"] == row["seed"]
                and d["identities"] == ids,
                "Invalid primary report",
            )
            require(
                (
                    d["partner"],
                    d["opponent"],
                    d["signals"],
                    d["card_policy"],
                    d["precision"],
                )
                == (partner, "native", "normal", conf["choice_policy"], v),
                "Wrong human primary condition",
            )
            config = d["configurations"]
            require(
                len(config) == 4
                and {(c["players"], c["target"], c["flor"]) for c in config}
                == {x for x in CONFIGURATIONS if x[0] == n},
                "Missing rules configuration",
            )
            for c in config:
                vals = c["pair_scores"]
                require(
                    c["pairs"] == 2000
                    and c["matches"] == 2000 * n
                    and len(vals) == 2000
                    and all(x in {i / n for i in range(n + 1)} for x in vals),
                    "Missing genuine all-seat paired deals",
                )
            per_precision[v, k] = np.mean([c["pair_scores"] for c in config], axis=0)
            if v == "int8":
                partners[k] = float(per_precision[v, k].mean())
    for n, goal in [(2, 0.70), (4, 0.60)]:
        keys = [k for k, v in expected.items() if v[0] == n]
        a = np.mean([per_precision["int8", k] for k in keys], axis=0)
        b = np.mean([per_precision["fp32", k] for k in keys], axis=0)
        modes[str(n)] = {"rate": float(a.mean()), "ci95": interval(a)}
        require(
            a.mean() >= goal and modes[str(n)]["ci95"][0] > 0.50,
            "Native primary target missed",
        )
        losses.append(b - a)
    loss = np.mean(losses, axis=0)
    ci = interval(loss)
    require(
        ci[1] <= 0.01,
        "Human-primary quantization loss upper bound exceeds one percentage point",
    )
    return {
        "modes": modes,
        "partners": partners,
        "quantization_loss": float(loss.mean()),
        "quantization_loss_ci95": ci,
        "seed": row["seed"],
        "pairs": row["pairs"],
    }


def evaluate_audit(root, project):
    base = root.parent
    conf, ident, res = check_inputs(root, project)
    m = load(root / "inputs/bundle/manifest.json")
    cp = conf["checkpoint_sha256"]
    model = conf["model_sha256"]
    jobs = conf["specifications"]
    reports = {}
    invalid = {}
    missing = []
    checks = []

    def check(name, fn):
        try:
            checks.append({"name": name, "passed": True, "detail": fn()})
        except (ValueError, KeyError, TypeError, OSError, AssertionError) as exc:
            checks.append({"name": name, "passed": False, "reason": str(exc)})

    for name, spec in jobs.items():
        if not (root / "receipts" / f"{name}.json").exists():
            missing.append(name)
            continue
        try:
            d = attest(root, name, ident)
            require(
                (
                    d["specification"],
                    d["checkpoint_sha256"],
                    d["wasm_sha256"],
                    d["evaluation_source_commit"],
                )
                == (spec, cp, conf["wasm_sha256"], conf["source_commit"]),
                "Wrong model, source or specification",
            )
            require(
                d["quantized_sha256"]
                == (model if spec["precision"] == "int8" else None),
                "Unexpected precision",
            )
            if spec["kind"] == "matches":
                require(
                    d["seed"] == res["seed"] and d["ablation"] == spec["ablation"],
                    "Wrong comparison or holdout",
                )
                configurations(
                    d,
                    {
                        x
                        for x in CONFIGURATIONS
                        if not spec["players"] or x[0] == spec["players"]
                    },
                )
                on = (
                    digest(root / spec["opponent_file"])
                    if "opponent_file" in spec
                    else spec["opponent"]
                )
                if "response_seed" in spec:
                    on = digest(
                        response_checkpoint(
                            root, spec["players"], spec["response_seed"], conf
                        )
                    )
                pn = (
                    digest(root / spec["partner_file"])
                    if "partner_file" in spec
                    else spec["partner"]
                )
                require(
                    (d["opponent"], d["partner"]) == (on, pn),
                    "Opponent or partner differs from frozen plan",
                )
                if spec["opponent"] == "population":
                    require(
                        d["opponent_styles"] == conf["opponent_styles"]
                        and d["opponent_schedule"] == conf["opponent_schedule"],
                        "Changed population schedule",
                    )
                for c in d["configurations"]:
                    require(
                        c["actions"].get("false_signal_claim", 0) == 0,
                        "Deployed CPU emitted a false signal",
                    )
            elif spec["kind"] == "selective-bluff":
                require(
                    conf.get("selective_opening") == OPENING_PROTOCOL
                    and d["seed"] == start_seed(res)
                    and d["matches"] == len(d["rows"]) == OPENING_MATCHES
                    and d["fixture"] == OPENING_PROTOCOL,
                    "Wrong selective opening receipt",
                )
            else:
                require(
                    d["seed"] == res["bluff_seed"]
                    and d["pairs"] == res["bluff_pairs"]
                    and d["fixture"] == "weak-opening-v1"
                    and d["opponent"] == spec["opponent"],
                    "Unexpected bluff fixture",
                )
            reports[name] = d
        except (OSError, KeyError, ValueError, TypeError) as exc:
            invalid[name] = str(exc)

    def report(n):
        if n not in reports:
            raise ValueError(f"Missing authenticated report {n}")
        return reports[n]

    def artifact():
        from .release_recurrent import BrowserActor

        BrowserActor(
            root / "inputs/candidate.pt",
            root / "inputs/bundle",
            root / "bin/libretruco_policy.dylib",
        )
        snapshot = project / "ml/runs/rmappo-retraining-v1/source/manifest.json"
        require(
            digest(snapshot) == m["training_snapshot_sha256"]
            and training_commit(snapshot, project) == m["training_source_commit"],
            "Training source provenance mismatch",
        )
        require(
            m["wasm_sha256"] == digest(root / "bin/retruco_policy.wasm")
            and m["sha256"] == model,
            "Runtime identity mismatch",
        )
        return {
            "checkpoint": cp,
            "model": model,
            "wasm": m["wasm_sha256"],
            "parameters": m["actor_parameters"],
            "bytes": m["bytes"],
            "training_commit": m["training_source_commit"],
            "exporter_commit": m["exporter_commit"],
            "game_commit": conf["simulator_commit"],
        }

    check("format4_artifact_identity_and_provenance", artifact)
    check(
        "authenticated_holdout_reports",
        lambda: (
            require(
                not invalid and not missing,
                f"{len(missing)} missing, {len(invalid)} invalid",
            )
            or {"expected": len(jobs), "verified": len(reports), "reservation": res}
        ),
    )
    check(
        "native_human_primary_and_paired_quantization",
        lambda: native_human_primary(base, project, conf),
    )
    check(
        "deployed_playing_strength",
        lambda: quality(report("quality-quantized"), cp, model),
    )
    check(
        "quantization_loss",
        lambda: quantization(
            report("quality-reference"), report("quality-quantized"), cp, model
        ),
    )

    def performance():
        profile = conf["choice_policy"].removeprefix("value-and-team-")
        p = load(base / f"performance-{profile}.json")
        require(
            p["casesSha256"] == digest(base / "performance-cases.json")
            and p.get("performancePass") is True,
            "Wrong performance cases or budget failure",
        )
        return browser_performance(p, model, m["wasm_sha256"])

    check("browser_performance", performance)

    def parity():
        r = project / "ml/runs" / conf.get("candidate_run", "reference-envido-v1")
        historical = load(r / "parity.json")
        web = load(base / "current-browser-parity.json")
        require(
            historical["checkpoint_sha256"] == cp
            and historical.get("choice_policy") == conf["choice_policy"]
            and web["model_sha256"] == model
            and web["wasm_sha256"] == m["wasm_sha256"],
            "Parity identity mismatch",
        )
        require(
            digest(r / "parity.evidence.json") == historical["evidence_sha256"]
            and digest(r / "parity.team-cases.json") == historical["team_cases_sha256"],
            "Changed frozen parity fixtures",
        )
        require(
            historical["native_sha256"] == conf["files"]["bin/libretruco_policy.dylib"]
            and web["team_decisions"] == historical["team_decisions"],
            "Wrong Rust runtime or differing team mappings",
        )
        e = load(r / "parity.evidence.json")
        results = {
            k: differences(e["quantized"], a)
            for k, a in [
                ("rust", e["native"]),
                ("node", web["node"]),
                ("browser", web["browser"]),
            ]
        }
        errors = {
            k: float(np.max(np.abs(np.asarray(e["quantized_states"]) - np.asarray(a))))
            for k, a in [
                ("rust", e["native_states"]),
                ("node", web["node_states"]),
                ("browser", web["browser_states"]),
            ]
        }
        require(
            len(e["quantized"]) == historical["cases"]
            and all(v["max_probability_error"] <= 1e-4 for v in results.values())
            and all(v <= 2e-4 for v in errors.values()),
            "Numerical or memory parity failed",
        )
        return {
            "cases": len(e["quantized"]),
            "runtime": results,
            "memory_errors": errors,
            "team_decisions": web["team_decisions"],
            "original_vs_int8": differences(e["float"], e["quantized"]),
        }

    check("candidate_runtime_parity", parity)

    def coverage():
        actor_seed = torch.load(
            root / "inputs/candidate.pt", weights_only=True, map_location="cpu"
        )["configuration"]["seed"]
        require(type(actor_seed) is int, "Missing candidate training seed")
        actors = {}
        partner_seeds = set()
        for label, role in [("historical", "historical"), ("crossplay", "partner")]:
            digests = set()
            for i in range(2):
                report(f"{label}-{i}")
                path = root / f"inputs/{role}-{i}.pt"
                other = load_checkpoint(path)
                ah = hashlib.sha256(serialize(other)[0]).hexdigest()
                require(
                    ah != model and digest(path) != cp, "Coverage repeats candidate"
                )
                digests.add(ah)
                if role == "partner":
                    saved = torch.load(path, weights_only=True, map_location="cpu")
                    s = saved.get("initial_seed", saved.get("config", {}).get("seed"))
                    require(
                        type(s) is int and s != actor_seed,
                        "Partner lacks independent training seed",
                    )
                    partner_seeds.add(s)
            require(len(digests) == 2, "Repeated frozen actor")
            actors[role] = sorted(digests)
        require(len(partner_seeds) == 2, "Repeated partner seed")
        scripted = {
            s: report("opponent-" + s)["pooled"] for s in conf["opponent_styles"]
        }
        partners = {
            s: report("partner-" + s)["pooled"]
            for s in ("literal", "cautious", "call-heavy", "silent", "ignore")
        }
        return {
            "scripted": scripted,
            "human_role": partners,
            "actors": actors,
            "independent_partner_seeds": sorted(partner_seeds),
        }

    check("opponent_and_partner_coverage", coverage)

    def ablation(ref, changed, mode):
        a, b = report(ref), report(changed)
        for k in (
            "opponent",
            "partner",
            "quantized_sha256",
            "opponent_schedule",
            "opponent_styles",
            "seed",
        ):
            require(a[k] == b[k], "Ablation differs in " + k)
        require(
            a["ablation"] == "none"
            and b["ablation"] == mode
            and a["quantized_sha256"] == model,
            "Wrong intervention",
        )
        return paired_loss(a, b)

    def benefit(ref, changed, mode):
        gain = ablation(ref, changed, mode)
        require(
            gain["cluster_ci95"][0] > 0,
            f"Benefit not statistically established: {gain['loss']}, CI {gain['cluster_ci95']}",
        )
        return gain

    check(
        "adaptation_benefit",
        lambda: benefit(
            "population-reference", "population-no-adaptation", "adaptation"
        ),
    )
    check(
        "signals_benefit",
        lambda: benefit("communication-literal", "no-communication-literal", "signals"),
    )

    def bluff():
        a = report("bluff-baseline")
        b = report("bluff-cautious")
        require(
            a["information_sha256"] == b["information_sha256"],
            "Inconsistent fixed initial information",
        )
        require(
            a["pairs"] >= 1000 and a["raw_rule_offers"] == a["pairs"],
            "Unexpected raw fixture",
        )
        if conf.get("selective_opening") == OPENING_PROTOCOL:
            require(
                all(
                    z["deployed_offers"] == 0
                    and z["raw_rule_offers"] == z["pairs"]
                    and z["policy_raise_probability"] == 0
                    for z in (a, b)
                ),
                "The weak fixed-hand veto failed",
            )
            return verify_ordinary(report("bluff-ordinary"), res)
        require(
            a["deployed_offers"] == a["pairs"]
            and 0 < a["policy_raise_probability"] <= 1,
            f"Deployed profile offered the required Truco in {a['deployed_offers']}/{a['pairs']} hidden deals; policy probability {a['policy_raise_probability']}",
        )
        gains = a["paired_win_gains"]
        require(
            len(gains) == a["pairs"] and all(x in (-1, 0, 1) for x in gains),
            "Invalid bluff pairs",
        )
        ci = interval(gains)
        require(ci[0] > 0, f"Conditional bluff gain not established: {ci}")
        return {
            "gain": float(np.mean(gains)),
            "ci95": ci,
            "cautious_diagnostic": b["win_gain_ci95"],
        }

    check("conditional_bluff_value", bluff)

    def best_responses():
        plan = load(root / "training-plan.json")
        details = []
        init_hashes = set()
        seen = set()
        for path, h in plan["training_source_files"].items():
            require(
                digest(root / "training-source" / path) == h,
                "Changed frozen response training source",
            )
        for n, s in plan["opponents"]:
            p = response_checkpoint(root, n, s, conf)
            d = report(f"response-{n}p-{s}")
            saved = torch.load(p, weights_only=True, map_location="cpu")
            require(
                saved["config"]["simulator_sha256"] == conf["files"]["bin/selfplay"]
                and saved["config"]["runtime_sha256"]
                == conf["files"]["bin/libretruco_policy.dylib"],
                "Target engine changed",
            )
            for name, h in saved["config"]["source_files"].items():
                require(
                    h == plan["training_source_files"]["ml/retruco_ml/" + name],
                    "Wrong training source",
                )
            trained_hash = hashlib.sha256(
                b"".join(v.numpy().tobytes() for v in saved["model"].values())
            ).hexdigest()
            require(
                trained_hash != saved["initial_model_sha256"]
                and digest(p) == d["opponent"],
                "Frozen or wrong response opponent",
            )
            init_hashes.add(saved["initial_model_sha256"])
            seen.add((n, s))
            details.append(
                {
                    "players": n,
                    "seed": s,
                    "training_seconds": saved["total_seconds"],
                    "updates": saved["update"],
                    "choices": saved["decisions"],
                    "checkpoint_sha256": digest(p),
                    "candidate_against_response": d["pooled"],
                }
            )
        require(
            len(init_hashes) == 6 and sorted(n for n, s in seen) == [2, 2, 2, 4, 4, 4],
            "Require six independent trained opponents",
        )
        return details

    check("empirical_best_responses", best_responses)

    def integration():
        d = load(base / "integration-merged.json")
        require(
            d["checkpoint_sha256"] == cp
            and d["wasm_sha256"] == m["wasm_sha256"]
            and d["soak_matches_per_mode"] >= 10
            and d["server_restarts"] >= 1,
            "Incomplete full-length recovery run",
        )
        names = integration_evidence(d, base, model, conf["simulator_commit"])
        for file, enabled in [
            ("production-worker-default.json", True),
            ("production-worker-off.json", False),
            ("preview-worker-default.json", False),
        ]:
            result = load(base / file)
            for row in result["rows"]:
                expected = (
                    "learned"
                    if enabled and row["simulatedManifestApproval"]
                    else "heuristic"
                )
                require(
                    row["decisionSources"] == [expected, expected],
                    "Incorrect emitted production policy",
                )
        return {
            "checks": names,
            "soak_matches_per_mode": d["soak_matches_per_mode"],
            "server_restarts": d["server_restarts"],
        }

    check("integration_and_recovery", integration)

    def human():
        h = load(base / "human-review.json")
        require(
            (h["checkpoint_sha256"], h["model_sha256"], h["wasm_sha256"])
            == (cp, model, m["wasm_sha256"])
            and h.get("choice_policy") == conf["choice_policy"]
            and set(h["played_modes"]) == {"1v1", "2v2"}
            and h["reviewer"]
            and h["notes"]
            and h["coordination_acceptable"] is True
            and h["weaknesses_reviewed"] is True
            and h["release_exception"] is False,
            "Missing exact-version human review",
        )
        return {
            "reviewer": h["reviewer"],
            "modes": h["played_modes"],
            "notes": h["notes"],
        }

    check("human_partnership_review", human)
    supplemental = {}
    for label, left, right, mode in [
        (
            "adaptation",
            "population-reference",
            "population-no-adaptation",
            "adaptation",
        ),
        (
            "signals-literal",
            "communication-literal",
            "no-communication-literal",
            "signals",
        ),
        (
            "signals-learned",
            "communication-learned",
            "no-communication-learned",
            "signals",
        ),
    ]:
        if left in reports and right in reports:
            supplemental[label] = ablation(left, right, mode)
    all_results = not missing and not invalid
    return {
        "ready": all_results and all(c["passed"] for c in checks),
        "complete_report_coverage": all_results,
        "expected_reports": len(jobs),
        "verified_reports": len(reports),
        "missing": missing,
        "invalid": invalid,
        "fingerprint": ident,
        "evaluation_commit": conf["source_commit"],
        "game_commit": conf["simulator_commit"],
        "reservation": res,
        "checks": checks,
        "paired_ablations": supplemental,
        "approval_changed": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = evaluate_audit(a.root, a.project)
    atomic_json(a.output, result)
    print(
        json.dumps(
            {
                "complete": result["complete_report_coverage"],
                "verified": result["verified_reports"],
                "ready": result["ready"],
                "unmet": [c["name"] for c in result["checks"] if not c["passed"]],
            }
        )
    )
    if not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
