from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def mutate_color(attrs: dict) -> dict:
    colors = ["red", "blue", "black", "white", "green", "yellow", "pink", "gray"]
    new_attrs = dict(attrs)
    if "color" in new_attrs:
        new_attrs["color"] = random.choice([c for c in colors if c != new_attrs["color"]])
    else:
        new_attrs["color"] = random.choice(colors)
    return new_attrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", default="data/processed/processed_rows.parquet")
    ap.add_argument("--out_parquet", default="data/processed/toy_noisy_rows.parquet")
    ap.add_argument("--copies_per_row", type=int, default=15)
    ap.add_argument("--noise_rate", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)

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
            rows.append(rr)

    toy = pd.DataFrame(rows).reset_index(drop=True)

    # Inject noise into a subset
    n = len(toy)
    m = int(n * args.noise_rate)
    idxs = random.sample(range(n), m)

    # For some rows: swap image_path with another row (Type I)
    # For others: change color attribute + canonical_text (Type I-ish)
    for idx in idxs:
        if random.random() < 0.5:
            # swap image with a random different row
            j = random.randrange(n)
            if j != idx:
                toy.at[idx, "image_path"], toy.at[j, "image_path"] = toy.at[j, "image_path"], toy.at[idx, "image_path"]
            toy.at[idx, "noise_label"] = "swap_image"
        else:
            # mutate color text/attributes
            attrs = json.loads(toy.at[idx, "attributes"])
            attrs2 = mutate_color(attrs)
            toy.at[idx, "attributes"] = json.dumps(attrs2, ensure_ascii=False)
            # rebuild canonical_text
            attrs_for_text = ", ".join([f"{k}={v}" for k, v in attrs2.items()])
            toy.at[idx, "canonical_text"] = f"{toy.at[idx,'title']}. Category: {toy.at[idx,'category']}. Attributes: {attrs_for_text}."
            toy.at[idx, "noise_label"] = "mutate_text"

    toy.to_parquet(out_p, index=False)
    print("✅ Wrote:", out_p)
    print("Rows:", len(toy))
    print("Noisy:", (toy["noise_label"] != "clean").sum())
    print(toy[["row_id", "category", "noise_label", "image_path"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
