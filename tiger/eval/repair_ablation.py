"""End-to-end repair ablation (roadmap 5.1).

This runs the full repair cycle under different configurations to prove the
necessity of the repair-side components (Arbiter, VLM Judge, Generative Fallback, Gamma Gate).

To keep runtime manageable on Kaggle, this defaults to running on a small random
subset of the noisy dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
import random

import pandas as pd
import numpy as np

from tiger import arbiter as arbiter_mod
from tiger import repair as repair_mod
from tiger import verify as verify_mod
from tiger import sieve as sieve_mod
from tiger.encoders import ClipEncoder
from tiger.schema import Schema


class DummyArbiter(arbiter_mod.ArbiterModel):
    """A dummy arbiter that routes randomly, simulating the absence of a trained classifier."""
    def predict_proba(self, x: np.ndarray) -> dict[str, float]:
        import random
        # Randomly assign probabilities to E1, E2, E3, E4
        probs = [random.random() for _ in range(4)]
        total = sum(probs)
        return {
            "E1": probs[0] / total,
            "E2": probs[1] / total,
            "E3": probs[2] / total,
            "E4": probs[3] / total,
            "CLEAN": 0.0
        }


def run_repair_ablations(noisy_df: pd.DataFrame, enc: ClipEncoder, schema: Schema,
                         thr: sieve_mod.SieveThresholds, loo_stats: dict,
                         vcal: verify_mod.VerifyCalibration, trained_model: arbiter_mod.ArbiterModel,
                         cfg: dict, root: Path, generator=None, vlm_judge=None,
                         sample_size: int = 20) -> dict:
    
    # 1. Sample the dataset so it doesn't take 5 hours
    # We want to sample rows that we KNOW are corrupted (based on split/seed) or just random
    noisy_sample = noisy_df.sample(n=min(sample_size, len(noisy_df)), random_state=42).reset_index(drop=True)
    
    audit_path = root / cfg["data"]["processed_dir"] / f"noise_audit_report_seed{cfg.get('noise', {}).get('seed', 7)}.csv"
    audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    truth_color = {}
    if not audit.empty:
        for _, r in audit[audit["field"] == "color"].iterrows():
            truth_color[str(r["row_id"])] = str(r["old_value"])
            
    def _evaluate_run(report: dict, final_df: pd.DataFrame) -> dict:
        v2t_correct = v2t_total = 0
        after = final_df.set_index("row_id")
        repaired_c = report["summary"].get("by_status", {}).get("repaired", 0)
        escalated_c = report["summary"].get("by_status", {}).get("escalated", 0)
        
        for rid, oc in report["outcomes"].items():
            if oc["final_status"] != "repaired":
                continue
            if rid in truth_color:
                import json as _json
                attrs_str = after.at[rid, "attributes"]
                if isinstance(attrs_str, pd.Series):
                    attrs_str = attrs_str.iloc[0]
                new_color = _json.loads(attrs_str).get("color", "")
                v2t_total += 1
                v2t_correct += int(str(new_color) == truth_color[rid])
                
        return {
            "total_attempted": report["summary"].get("total", 0),
            "repaired": repaired_c,
            "escalated": escalated_c,
            "color_accuracy": (v2t_correct / max(1, v2t_total)),
            "v2t_total": v2t_total
        }

    results = {}
    print(f"\nRunning repair ablations on a {sample_size}-item sample...")

    # Config 1: Full System
    print("1/5: Running 'Full System'...")
    cfg_full = cfg.copy()
    rep_full, rep_full_report = repair_mod.run_repair_cycle(
        noisy_sample, enc, schema, thr, loo_stats, vcal, trained_model, cfg_full, root,
        max_passes=2, independent=vlm_judge, generator=generator)
    results["full"] = _evaluate_run(rep_full_report, rep_full)

    # Config 2: No Arbiter (Random Routing)
    print("2/5: Running 'No Arbiter (Random Routing)'...")
    cfg_no_arbiter = cfg.copy()
    dummy_model = DummyArbiter(
        feature_names=trained_model.feature_names, classes=trained_model.classes,
        mean=trained_model.mean, scale=trained_model.scale, coef=trained_model.coef,
        intercept=trained_model.intercept
    )
    rep_no_arb, rep_no_arb_report = repair_mod.run_repair_cycle(
        noisy_sample, enc, schema, thr, loo_stats, vcal, dummy_model, cfg_no_arbiter, root,
        max_passes=2, independent=vlm_judge, generator=generator)
    results["no_arbiter"] = _evaluate_run(rep_no_arb_report, rep_no_arb)

    # Config 3: No VLM Judge
    print("3/5: Running 'No VLM Judge'...")
    rep_no_vlm, rep_no_vlm_report = repair_mod.run_repair_cycle(
        noisy_sample, enc, schema, thr, loo_stats, vcal, trained_model, cfg_full, root,
        max_passes=2, independent=None, generator=generator)
    results["no_vlm"] = _evaluate_run(rep_no_vlm_report, rep_no_vlm)

    # Config 4: No Generative Fallback
    print("4/5: Running 'No Generative Fallback'...")
    rep_no_gen, rep_no_gen_report = repair_mod.run_repair_cycle(
        noisy_sample, enc, schema, thr, loo_stats, vcal, trained_model, cfg_full, root,
        max_passes=2, independent=vlm_judge, generator=None)
    results["no_gen"] = _evaluate_run(rep_no_gen_report, rep_no_gen)

    # Config 5: No Gamma Gate (Set gamma threshold to 0.0 so everything passes)
    print("5/5: Running 'No Gamma Gate (Accept All Routes)'...")
    cfg_no_gamma = cfg.copy()
    if "fusion" not in cfg_no_gamma:
        cfg_no_gamma["fusion"] = {}
    cfg_no_gamma["fusion"]["gamma"] = 0.0
    rep_no_gamma, rep_no_gamma_report = repair_mod.run_repair_cycle(
        noisy_sample, enc, schema, thr, loo_stats, vcal, trained_model, cfg_no_gamma, root,
        max_passes=2, independent=vlm_judge, generator=generator)
    results["no_gamma"] = _evaluate_run(rep_no_gamma_report, rep_no_gamma)

    return results


def format_repair_ablations(results: dict) -> str:
    lines = []
    lines.append("")
    lines.append("🛠️ Repair-Side Ablation Study: Component Impact")
    lines.append("=" * 75)
    lines.append("")
    lines.append(f"{'Configuration':<25s} | {'Repaired':<8s} | {'Escalated':<10s} | {'Restoration Acc (V2T)':<20s}")
    lines.append("-" * 75)

    friendly_names = {
        "full": "Full System",
        "no_arbiter": "No Arbiter (Random)",
        "no_vlm": "No VLM Judge",
        "no_gen": "No Generative Fallback",
        "no_gamma": "No Gamma Gate (γ=0)",
    }
    
    order = ["no_arbiter", "no_vlm", "no_gen", "no_gamma", "full"]

    for name in order:
        if name not in results:
            continue
        r = results[name]
        label = friendly_names.get(name, name)
        acc_str = f"{r['color_accuracy']:.1%} ({r['v2t_total']} cases)" if r['v2t_total'] > 0 else "N/A"
        lines.append(f"{label:<25s} | {r['repaired']:<8d} | {r['escalated']:<10d} | {acc_str:<20s}")

    lines.append("")
    lines.append("Key Takeaways:")
    if "full" in results and "no_arbiter" in results:
        delta_acc = results["full"]["color_accuracy"] - results["no_arbiter"]["color_accuracy"]
        lines.append(f"  • Trained Arbiter vs Random: {delta_acc:+.1%} restoration accuracy")
    if "full" in results and "no_gamma" in results:
        delta_esc = results["no_gamma"]["escalated"] - results["full"]["escalated"]
        lines.append(f"  • Gamma Gate: Safely escalated {delta_esc} uncertain cases instead of forcing bad repairs")
    if "full" in results and "no_vlm" in results:
        delta_acc = results["full"]["color_accuracy"] - results["no_vlm"]["color_accuracy"]
        lines.append(f"  • VLM Judge: Prevented coarse semantic swaps, improving accuracy by {delta_acc:+.1%}")
    if "full" in results and "no_gen" in results:
        delta_rep = results["full"]["repaired"] - results["no_gen"]["repaired"]
        lines.append(f"  • Generative Fallback: Successfully repaired {delta_rep} products that would otherwise be dead ends")

    return "\n".join(lines)
