"""What the dismiss guard costs, swept offline (D14). Not a pytest module.

Run:  python tests/bench_dismiss_guard.py [path/to/run/outputs]

The Sieve flags a row when a probe z-margin reaches `sieve.probes.z_margin`
(2.0, i.e. z <= -2.0). The Arbiter then refuses to dismiss a CLEAN verdict
while any probe z <= -2.0. Those are the same test on the same quantity, so a
probe-flagged row can never be cleared -- the guard cancels the path it guards.

Dismissal is a pure routing decision over evidence already on disk, so the
trade can be measured without re-encoding anything. This sweeps the guard
threshold and the dismiss threshold on the CALIBRATION seeds (never the
reporting split) and reports both sides of the trade:

  cleared     -- clean rows dismissed instead of sent to a human. The win.
  leaked      -- dirty rows dismissed. The cost, and it is the dangerous one:
                 a dismissed dirty row is silently kept in the catalogue.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from tiger import arbiter as A
from tiger import text_views
from tiger.schema import load_schema
from tiger.sieve import _title_color

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = ROOT / "data/outputs"
SEEDS = [1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014]
GUARD_Z = [-2.0, -2.5, -3.0, -3.5, -4.0, -99.0]     # -99 = guard disabled
DISMISS_P = [0.70, 0.80, 0.90]
SHIPPED_GUARD_Z = -3.0      # configs/tiger.yaml arbiter.dismiss_contrary_z


def load(outputs: Path, seed: int, recompute_title: bool = False):
    """Evidence + ground-truth labels for one calibration seed.

    `recompute_title` re-derives `title_contradiction` from the row's title with
    the *current* `_title_color`, rather than using the value frozen into the
    evidence file by the run. The flag is a pure text check, so this predicts
    what the next run will do without re-encoding anything (D15).
    """
    ev = [json.loads(l) for l in (outputs / f"evidence_cal_seed{seed}.jsonl").open()]
    sieve = pd.read_parquet(outputs / f"sieve_cal_seed{seed}.parquet").set_index("row_id")
    truth = sieve["noise_label"].astype(str).to_dict()

    if recompute_title:
        schema = load_schema(ROOT / "configs/schema.yaml")
        for e in ev:
            rid = e["row_id"]
            if rid not in sieve.index:
                continue
            attrs = text_views.parse_attrs(sieve.at[rid, "attributes"])
            declared = schema.normalize("color", attrs.get("color", "")) if attrs.get("color") else ""
            tc = _title_color(str(sieve.at[rid, "title"]), schema, str(attrs.get("brand", "")))
            e["title_contradiction"] = bool(tc and declared and tc != declared)

    lab = [truth.get(e["row_id"], "clean") for e in ev]
    keep = [(e, l) for e, l in zip(ev, lab)
            if not e.get("image_missing") and not e.get("text_missing")]
    return [e for e, _ in keep], [l for _, l in keep]


def main() -> None:
    outputs = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUTS
    recompute = "--recompute-title" in sys.argv
    model = A.ArbiterModel.from_json(
        (ROOT / "paper_assets/results/abo/thresholds/tiger_arbiter_model.json").read_text())

    evs, labs = [], []
    for s in SEEDS:
        if not (outputs / f"evidence_cal_seed{s}.jsonl").exists():
            continue
        e, l = load(outputs, s, recompute_title=recompute)
        evs += e
        labs += l
    if not evs:
        raise SystemExit(f"no calibration evidence under {outputs}")
    if recompute:
        print("title_contradiction RECOMPUTED with the current text check (D15)\n")

    n_clean = sum(1 for l in labs if l == "clean")
    n_dirty = len(labs) - n_clean
    print(f"{len(evs)} flagged rows from {len(SEEDS)} calibration seeds: "
          f"{n_clean} clean, {n_dirty} dirty\n")
    print(f"{'guard z':>8s} {'dismiss p':>10s} {'cleared':>18s} {'leaked (dirty)':>18s} "
          f"{'precision':>10s}")
    print("-" * 70)

    for gz in GUARD_Z:
        for dp in DISMISS_P:
            cfg = {"arbiter": {"gamma": 0.60, "dismiss_threshold": dp,
                               "dismiss_contrary_z": gz,
                               "t2v_policy": {}}}
            cleared = leaked = 0
            for ev, lab in zip(evs, labs):
                if A.route(ev, model, cfg).action != "dismiss":
                    continue
                if lab == "clean":
                    cleared += 1
                else:
                    leaked += 1
            total = cleared + leaked
            prec = cleared / total if total else float("nan")
            tag = "  <- shipped" if (gz == SHIPPED_GUARD_Z and dp == 0.80) else ""
            gz_s = "off" if gz < -50 else f"{gz:.1f}"
            print(f"{gz_s:>8s} {dp:>10.2f} {cleared:>7d} ({cleared / n_clean:>5.1%}) "
                  f"{leaked:>7d} ({leaked / n_dirty:>5.1%}) {prec:>10.3f}{tag}")


if __name__ == "__main__":
    main()
