from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


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
    # Replace whole-word color tokens (case-insensitive)
    pat = re.compile(rf"\b{re.escape(old_color)}\b", re.IGNORECASE)
    return pat.sub(new_color, text)


def maybe_replace_any_color_in_title(title: str, new_color: str) -> tuple[str, str | None]:
    """
    If title contains any known color word, replace the first match with new_color.
    Returns (new_title, old_color_found_or_None)
    """
    lower = title.lower()
    for c in COLOR_VOCAB:
        if re.search(rf"\b{re.escape(c)}\b", lower):
            return replace_color_in_text(title, c, new_color), c
    return title, None


def build_canonical(title: str, category: str, attrs: dict) -> str:
    # Make attributes (especially color) prominent and consistent.
    title = str(title) if title is not None else ""
    category = str(category) if category is not None else ""
    color = attrs.get("color", None)

    # Stable ordering for deterministic text
    items = sorted((str(k), str(v)) for k, v in attrs.items())
    attrs_for_text = ", ".join([f"{k}={v}" for k, v in items])

    color_prefix = f"Color: {color}. " if color else ""
    return f"{title}. {color_prefix}Category: {category}. Attributes: {attrs_for_text}."


def mutate_text_color_flip(rng: random.Random, title: str, category: str, attrs: dict, strength: int):
    """
    strength 1: flip attrs.color only (canonical includes it prominently)
    strength 2: also replace color in title if present
    strength 3: introduce contradiction (title + attrs disagree)
    """
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
        # Try to replace any color word in title; if none found, append a short tag
        title2, found = maybe_replace_any_color_in_title(str(title), new)
        if found is None:
            title2 = f"{title} ({new})"
        canon = build_canonical(title2, category, attrs2)
        return title2, attrs2, canon, noise_subtype

    # strength 3: contradiction: title says old (or another), attrs says new
    attrs2 = dict(attrs)
    attrs2["color"] = new
    title2 = str(title)
    # Force title to contain a different color word if possible
    if old and old != new:
        # If title has any color, replace it with old; else append old
        t3, found = maybe_replace_any_color_in_title(title2, old)
        title2 = t3 if found else f"{title2} ({old})"
    else:
        # if old missing, pick a different color for title
        title_color = pick_different(rng, COLOR_VOCAB, new)
        t3, found = maybe_replace_any_color_in_title(title2, title_color)
        title2 = t3 if found else f"{title2} ({title_color})"

    canon = build_canonical(title2, category, attrs2)
    return title2, attrs2, canon, noise_subtype


def mutate_text_contradictory_phrase(rng: random.Random, title: str, category: str, attrs: dict, strength: int):
    """
    Contradiction via text phrase injection.
    strength 1: append a wrong 'Color:' sentence to canonical only (attrs unchanged)
    strength 2: also inject into title
    strength 3: keep two conflicting colors inside canonical (very strong)
    """
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
        # add wrong color hint to title too
        title2 = f"{title2} in {wrong}"
        canon = build_canonical(title2, category, attrs2) + f" Note: Color: {wrong}."
        return title2, attrs2, canon, noise_subtype

    # strength 3: two conflicting statements in canonical (very strong)
    canon = base + f" Note: Color: {wrong}. Also described as Color: {stated or 'unknown'}."
    return title2, attrs2, canon, noise_subtype


def mutate_text_attribute_drop(rng: random.Random, title: str, category: str, attrs: dict, strength: int):
    """
    Drop/erase an attribute (color) and optionally inject a wrong one.
    strength 1: remove attrs.color (canonical loses strong cue)
    strength 2: remove attrs.color but keep/add a color token in title (mismatch-ish)
    strength 3: remove attrs.color and inject wrong color explicitly in canonical
    """
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
        # Ensure title contains a color word
        t2, found = maybe_replace_any_color_in_title(title2, wrong)
        title2 = t2 if found else f"{title2} ({wrong})"
        canon = build_canonical(title2, category, attrs2)
        return title2, attrs2, canon, noise_subtype

    wrong = pick_different(rng, COLOR_VOCAB, old)
    canon = build_canonical(title2, category, attrs2) + f" Color: {wrong}."
    return title2, attrs2, canon, noise_subtype


def mutate_text_object_flip(rng: random.Random, title: str, category: str, attrs: dict, strength: int):
    """
    Flip an object term in title/canonical. If no known term found, fallback to strong contradiction.
    strength 1: attempt replacement once
    strength 2: replace + append object hint
    strength 3: force object hint strongly (even if not found)
    """
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
        # fallback: add a strong object claim to canonical (still hurts similarity)
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
    """
    Choose a subtype and apply it. Returns updated (title, attrs_dict, canonical_text, subtype).
    """
    title = str(row.get("title", ""))
    category = str(row.get("category", ""))
    attrs = _safe_json_load(row.get("attributes"))

    # You can tune these weights
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
    ap.add_argument("--in_parquet", default="data/processed/processed_rows.parquet")
    ap.add_argument("--out_parquet", default="data/processed/toy_noisy_rows.parquet")
    ap.add_argument("--copies_per_row", type=int, default=15)
    ap.add_argument("--noise_rate", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--p_swap_image", type=float, default=0.50)
    ap.add_argument("--max_strength", type=int, default=3)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    root = Path(__file__).resolve().parents[1]
    in_p = (root / args.in_parquet).resolve()
    out_p = (root / args.out_parquet).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_p)

    # Duplicate rows to create enough samples per category
    rows = []
    for _, r in df.iterrows():
        for k in range(args.copies_per_row):
            rr = r.copy()
            rr["row_id"] = f"{r['row_id']}_{k}"
            rr["noise_label"] = "clean"
            rr["noise_subtype"] = "clean"
            rr["noise_strength"] = 0
            rows.append(rr)

    toy = pd.DataFrame(rows).reset_index(drop=True)

    # Inject noise into a subset
    n = len(toy)
    m = int(n * args.noise_rate)
    idxs = rng.sample(range(n), m)

    for idx in idxs:
        if rng.random() < args.p_swap_image:
            j = rng.randrange(n)
            if j != idx:
                # copy full image-related fields so the row looks consistent
                for col in ["image_path", "source_image_path", "source_image_url", "is_image_missing"]:
                    if col in toy.columns:
                        toy.at[idx, col] = toy.at[j, col]

            toy.at[idx, "noise_label"] = "swap_image"
            toy.at[idx, "noise_subtype"] = "swap_image"
            toy.at[idx, "noise_strength"] = 1
        else:
            strength = rng.randint(1, max(1, int(args.max_strength)))

            # mutate text/attributes/canonical
            title2, attrs2, canon2, subtype = apply_mutate_text(rng, toy.loc[idx], strength)

            toy.at[idx, "title"] = title2
            toy.at[idx, "attributes"] = _safe_json_dump(attrs2)
            toy.at[idx, "canonical_text"] = canon2

            toy.at[idx, "noise_label"] = "mutate_text"
            toy.at[idx, "noise_subtype"] = subtype
            toy.at[idx, "noise_strength"] = strength

    toy.to_parquet(out_p, index=False)

    print("✅ Wrote:", out_p)
    print("Rows:", len(toy))
    print("Noisy:", (toy["noise_label"] != "clean").sum())
    print("\nNoise label counts:\n", toy["noise_label"].value_counts(dropna=False).to_string())
    print("\nMutate_text subtype counts:\n", toy.loc[toy["noise_label"] == "mutate_text", "noise_subtype"].value_counts(dropna=False).to_string())
    print("\nMutate_text strength counts:\n", toy.loc[toy["noise_label"] == "mutate_text", "noise_strength"].value_counts(dropna=False).sort_index().to_string())
    print("\nSample rows:\n", toy[["row_id", "category", "noise_label", "noise_subtype", "noise_strength", "image_path"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
