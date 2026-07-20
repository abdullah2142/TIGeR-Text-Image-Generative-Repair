"""Encoder selection by per-field probe accuracy (roadmap 3.3).

The review is explicit that encoders must be chosen by fine-grained attribute
sensitivity, not global similarity: "Selection by per-field probe accuracy on
the labelled synthetic set, not by global similarity. Keep
openai/clip-vit-base-patch32 as the reported baseline."

For a given encoder and the CLEAN catalogue (declared value == true value), we
score each image against one ensembled caption per candidate value in Omega_j
and check whether the encoder's argmax equals the true value. Accuracy per field
is the discrimination metric; a better encoder raises it, which is what would
move the material-probe ceiling on real photos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tiger import text_views
from tiger.encoders import ClipEncoder
from tiger.schema import Schema


def probe_accuracy(encoder: ClipEncoder, df_clean: pd.DataFrame, schema: Schema,
                   fields: list[str]) -> dict:
    """Per-field top-1 probe accuracy of `encoder` on the clean catalogue."""
    df = df_clean.reset_index(drop=True)
    attrs_all = [text_views.parse_attrs(a) for a in df["attributes"]]

    paths = [str(p) for p in df["image_path"]]
    img_emb, ok = encoder.encode_images(paths)

    out: dict[str, dict] = {}
    for fld in fields:
        domain = schema.domain(fld)
        correct = total = 0
        per_cat_cand: dict[str, np.ndarray] = {}
        for cat in df["category"].astype(str).unique():
            embs = []
            for v in domain:
                e = encoder.encode_texts(text_views.field_caption_templates(cat, fld, v)).mean(axis=0)
                embs.append(e / (np.linalg.norm(e) + 1e-12))
            per_cat_cand[cat] = np.stack(embs)

        for i in df.index:
            if not ok[i]:
                continue
            declared = schema.normalize(fld, attrs_all[i].get(fld, "")) if attrs_all[i].get(fld) else ""
            if not declared or declared not in domain:
                continue
            cat = str(df.at[i, "category"])
            scores = per_cat_cand[cat] @ img_emb[i]
            pred = domain[int(np.argmax(scores))]
            total += 1
            correct += int(pred == declared)
        out[fld] = {"accuracy": correct / total if total else float("nan"),
                    "correct": correct, "total": total, "n_values": len(domain)}
    encoder.save_cache()
    return out


def compare(encoders: dict[str, ClipEncoder], df_clean: pd.DataFrame, schema: Schema,
            fields: list[str]) -> dict:
    return {name: probe_accuracy(enc, df_clean, schema, fields) for name, enc in encoders.items()}


def format_comparison(results: dict, fields: list[str]) -> str:
    names = list(results)
    header = f"{'field':10s} " + "  ".join(f"{n[:22]:>22s}" for n in names)
    lines = [header, "-" * len(header)]
    for fld in fields:
        cells = []
        for n in names:
            r = results[n].get(fld, {})
            cells.append(f"{r.get('accuracy', float('nan')):>16.3f} ({r.get('total', 0):>3d})")
        lines.append(f"{fld:10s} " + "  ".join(f"{c:>22s}" for c in cells))
    return "\n".join(lines)
