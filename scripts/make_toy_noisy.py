from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


COLOR_VOCAB = ["red", "blue", "black", "white", "green", "yellow", "pink", "gray", "purple", "brown", "orange"]
OBJECT_SWAPS = [
    ("shirt", "shoe"),
    ("shoes", "shirt"),
    ("jacket", "bag"),
    ("bag", "jacket"),
    ("dress", "pants"),
    ("pants", "dress"),
    ("hat", "shoe"),
]


# CLI defaults (used to detect whether user explicitly overrode)
CLI_DEFAULTS = {
    "copies_per_row": 15,
    "noise_rate": 0.30,
    "seed": 7,
    "p_swap_image": 0.50,
    "max_strength": 3,
}


def load_noise_config(project_root: Path, config_rel: str) -> dict:
    cfg_path = (project_root / config_rel).resolve()
    if not cfg_path.exists():
        return {}
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        return cfg.get("noise", {}) or {}
    except Exception:
        return {}


def get_git_commit(project_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or None
    except Exception:
        return None


def _safe_json_load(x):
    if isinstance(x, dict):
        return dict(x)
    if x is None:
        return {}
    if isinstance(x, float) and pd.isna(x):
        return {}
    s = str(x)
    if not s.strip():
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def _safe_json_dump(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False)


def pick_different(rng: random.Random, vocab: list[str], current: str | None) -> str:
    if current and current in vocab:
        choices = [v for v in vocab if v != current]
        return rng.choice(choices) if choices else current
    return rng.choice(vocab)


def replace_color_in_text(text: str, old_color: str, new_color: str) -> str:
    pat = re.compile(rf"\b{re.escape(old_color)}\b", re.IGNORECASE)
    return pat.sub(new_color, text)


def maybe_replace_any_color_in_title(title: str, new_color: str) -> tuple[str, str | None]:
    lower = title.lower()
    for c in COLOR_VOCAB:
        if re.search(rf"\b{re.escape(c)}\b", lower):
            return replace_color_in_text(title, c, new_color), c
    return title, None


def build_canonical(title: str, category: str, attrs: dict) -> str:
    title = str(title) if title is not None else ""
    category = str(category) if category is not None else ""
    color = attrs.get("color", None)

    items = sorted((str(k), str(v)) for k, v in attrs.items())
    attrs_for_text = ", ".join([f"{k}={v}" for k, v in items])

    color_prefix = f"Color: {color}. " if color else ""
    return f"{title}. {color_prefix}Category: {category}. Attributes: {attrs_for_text}."


def mutate_text_color_flip(rng: random.Random, title: str, category: str, attrs: dict, strength: int):
    old = str(attrs.get("color", "")).lower().strip() or None
    new = pick_different(rng, COLOR_VOCAB, old)

    noise_subtype = "color_flip"
    if strength == 1:
        attrs2 = dict(attrs)
        attrs2["color"] = new
        title2 = title
        canon = build_canonical(title2, category, attrs2)
        return title2, attrs2, canon, noise_subtype

    if strength == 2:
        attrs2 = dict(attrs)
        attrs2["color"] = new
        title2, found = maybe_replace_any_color_in_title(str(title), new)
        if found is None:
            title2 = f"{title} ({new})"
        canon = build_canonical(title2, category, attrs2)
        return title2, attrs2, canon, noise_subtype

    attrs2 = dict(attrs)
    attrs2["color"] = new
    title2 = str(title)
    if old and old != new:
        t3, found = maybe_replace_any_color_in_title(title2, old)
        title2 = t3 if found else f"{title2} ({old})"
    else:
        title_color = pick_different(rng, COLOR_VOCAB, new)
        t3, found = maybe_replace_any_color_in_title(title2, title_color)
        title2 = t3 if found else f"{title2} ({title_color})"

    canon = build_canonical(title2, category, attrs2)
    return title2, attrs2, canon, noise_subtype


def mutate_text_contradictory_phrase(rng: random.Random, title: str, category: str, attrs: dict, strength: int):
    attrs2 = dict(attrs)
    stated = str(attrs2.get("color", "")).lower().strip() or None
    wrong = pick_different(rng, COLOR_VOCAB, stated)

    noise_subtype = "contradictory_phrase"
    title2 = str(title)

    base = build_canonical(title2, category, attrs2)

    if strength == 1:
        canon = base + f" Note: Color: {wrong}."
        return title2, attrs2, canon, noise_subtype

    if strength == 2:
        title2 = f"{title2} in {wrong}"
        canon = build_canonical(title2, category, attrs2) + f" Note: Color: {wrong}."
        return title2, attrs2, canon, noise_subtype

    canon = base + f" Note: Color: {wrong}. Also described as Color: {stated or 'unknown'}."
    return title2, attrs2, canon, noise_subtype


def mutate_text_attribute_drop(rng: random.Random, title: str, category: str, attrs: dict, strength: int):
    attrs2 = dict(attrs)
    old = str(attrs2.get("color", "")).lower().strip() or None
    attrs2.pop("color", None)

    noise_subtype = "attribute_drop"
    title2 = str(title)

    if strength == 1:
        canon = build_canonical(title2, category, attrs2)
        return title2, attrs2, canon, noise_subtype

    if strength == 2:
        wrong = pick_different(rng, COLOR_VOCAB, old)
        t2, found = maybe_replace_any_color_in_title(title2, wrong)
        title2 = t2 if found else f"{title2} ({wrong})"
        canon = build_canonical(title2, category, attrs2)
        return title2, attrs2, canon, noise_subtype

    wrong = pick_different(rng, COLOR_VOCAB, old)
    canon = build_canonical(title2, category, attrs2) + f" Color: {wrong}."
    return title2, attrs2, canon, noise_subtype


def mutate_text_object_flip(rng: random.Random, title: str, category: str, attrs: dict, strength: int):
    noise_subtype = "object_flip"
    title2 = str(title)
    lower = title2.lower()

    replaced = False
    for a, b in OBJECT_SWAPS:
        if re.search(rf"\b{re.escape(a)}\b", lower):
            title2 = re.sub(rf"\b{re.escape(a)}\b", b, title2, flags=re.IGNORECASE)
            replaced = True
            break

    attrs2 = dict(attrs)

    if not replaced and strength >= 2:
        base = build_canonical(title2, category, attrs2)
        canon = base + " Object: shoe." if strength == 2 else base + " Object: shoe. Not a shirt."
        return title2, attrs2, canon, noise_subtype

    canon = build_canonical(title2, category, attrs2)
    if strength == 2:
        canon += " Object: shoe."
    elif strength == 3:
        canon += " Object: shoe. Not a shirt."
    return title2, attrs2, canon, noise_subtype


def apply_mutate_text(rng: random.Random, row, strength: int):
    title = str(row.get("title", ""))
    category = str(row.get("category", ""))
    attrs = _safe_json_load(row.get("attributes"))

    subtypes = [
        ("color_flip", 0.50),
        ("contradictory_phrase", 0.25),
        ("attribute_drop", 0.20),
        ("object_flip", 0.05),
    ]

    r = rng.random()
    acc = 0.0
    chosen = "color_flip"
    for name, w in subtypes:
        acc += w
        if r <= acc:
            chosen = name
            break

    if chosen == "color_flip":
        return mutate_text_color_flip(rng, title, category, attrs, strength)
    if chosen == "contradictory_phrase":
        return mutate_text_contradictory_phrase(rng, title, category, attrs, strength)
    if chosen == "attribute_drop":
        return mutate_text_attribute_drop(rng, title, category, attrs, strength)
    return mutate_text_object_flip(rng, title, category, attrs, strength)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--in_parquet", default="data/processed/processed_rows.parquet")
    ap.add_argument("--out_parquet", default="data/processed/toy_noisy_rows.parquet")
    ap.add_argument("--copies_per_row", type=int, default=CLI_DEFAULTS["copies_per_row"])
    ap.add_argument("--noise_rate", type=float, default=CLI_DEFAULTS["noise_rate"])
    ap.add_argument("--seed", type=int, default=CLI_DEFAULTS["seed"])
    ap.add_argument("--p_swap_image", type=float, default=CLI_DEFAULTS["p_swap_image"])
    ap.add_argument("--max_strength", type=int, default=CLI_DEFAULTS["max_strength"])
    ap.add_argument("--no_meta", action="store_true", help="Disable writing sidecar meta JSON")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]

    # -------- Upgrade 2C: load noise settings from config.yaml (determinism) --------
    noise_cfg = load_noise_config(root, args.config)

    def _pick(name: str, cfg_key: str, caster):
        """Use config value only if CLI arg equals its default (i.e., user likely didn't override)."""
        cli_val = getattr(args, name)
        if cli_val != CLI_DEFAULTS[name]:
            return cli_val
        if cfg_key in noise_cfg:
            try:
                return caster(noise_cfg[cfg_key])
            except Exception:
                return cli_val
        return cli_val

    seed = _pick("seed", "seed", int)
    copies_per_row = _pick("copies_per_row", "copies_per_row", int)
    noise_rate = _pick("noise_rate", "noise_rate", float)
    max_strength = _pick("max_strength", "max_strength", int)

    # p_swap_image supports either p_swap_image or rate_swap_image/rate_mutate_text in config
    if args.p_swap_image != CLI_DEFAULTS["p_swap_image"]:
        p_swap_image = float(args.p_swap_image)
    elif "p_swap_image" in noise_cfg:
        p_swap_image = float(noise_cfg["p_swap_image"])
    elif "rate_swap_image" in noise_cfg and "rate_mutate_text" in noise_cfg:
        rs = float(noise_cfg["rate_swap_image"])
        rm = float(noise_cfg["rate_mutate_text"])
        p_swap_image = rs / (rs + rm) if (rs + rm) > 0 else float(args.p_swap_image)
    else:
        p_swap_image = float(args.p_swap_image)

    rng = random.Random(seed)
    # -------------------------------------------------------------------------------

    in_p = (root / args.in_parquet).resolve()
    out_p = (root / args.out_parquet).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_p)

    rows = []
    for _, r in df.iterrows():
        for k in range(copies_per_row):
            rr = r.copy()
            rr["row_id"] = f"{r['row_id']}_{k}"
            rr["noise_label"] = "clean"
            rr["noise_subtype"] = "clean"
            rr["noise_strength"] = 0
            rows.append(rr)

    toy = pd.DataFrame(rows).reset_index(drop=True)

    n = len(toy)
    m = int(n * noise_rate)
    idxs = rng.sample(range(n), m)

    for idx in idxs:
        if rng.random() < p_swap_image:
            j = rng.randrange(n)
            if j != idx:
                for col in ["image_path", "source_image_path", "source_image_url", "is_image_missing"]:
                    if col in toy.columns:
                        toy.at[idx, col] = toy.at[j, col]

            toy.at[idx, "noise_label"] = "swap_image"
            toy.at[idx, "noise_subtype"] = "swap_image"
            toy.at[idx, "noise_strength"] = 1
        else:
            strength = rng.randint(1, max(1, int(max_strength)))

            title2, attrs2, canon2, subtype = apply_mutate_text(rng, toy.loc[idx], strength)

            toy.at[idx, "title"] = title2
            toy.at[idx, "attributes"] = _safe_json_dump(attrs2)
            toy.at[idx, "canonical_text"] = canon2

            toy.at[idx, "noise_label"] = "mutate_text"
            toy.at[idx, "noise_subtype"] = subtype
            toy.at[idx, "noise_strength"] = strength

    toy.to_parquet(out_p, index=False)

    # -------- Sidecar metadata (reproducibility) --------
    if not args.no_meta:
        meta_path = out_p.parent / f"{out_p.stem}.meta.json"
        meta = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(root),
            "in_parquet": str(in_p),
            "out_parquet": str(out_p),
            "noise_params": {
                "seed": seed,
                "copies_per_row": copies_per_row,
                "noise_rate": noise_rate,
                "p_swap_image": p_swap_image,
                "max_strength": max_strength,
            },
            "mutate_text_subtype_weights": [
                ["color_flip", 0.50],
                ["contradictory_phrase", 0.25],
                ["attribute_drop", 0.20],
                ["object_flip", 0.05],
            ],
            "vocab": {"colors": COLOR_VOCAB, "object_swaps": OBJECT_SWAPS},
            "counts": {
                "rows_total": int(len(toy)),
                "rows_noisy": int((toy["noise_label"] != "clean").sum()),
            },
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    # ---------------------------------------------------

    print("✅ Wrote:", out_p)
    print("Rows:", len(toy))
    print("Noisy:", (toy["noise_label"] != "clean").sum())
    print("\nNoise label counts:\n", toy["noise_label"].value_counts(dropna=False).to_string())
    print("\nMutate_text subtype counts:\n", toy.loc[toy["noise_label"] == "mutate_text", "noise_subtype"].value_counts(dropna=False).to_string())
    print("\nMutate_text strength counts:\n", toy.loc[toy["noise_label"] == "mutate_text", "noise_strength"].value_counts(dropna=False).sort_index().to_string())
    print("\nSample rows:\n", toy[["row_id", "category", "noise_label", "noise_subtype", "noise_strength", "image_path"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
