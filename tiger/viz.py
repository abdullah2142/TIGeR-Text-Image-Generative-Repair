"""Qualitative figures for the repair pipeline.

`build_qualitative_grid` is the one the paper uses; it draws with PIL only, so
it has no plotting dependency and runs wherever the pipeline runs.
`plot_repair_stages` is the older matplotlib version kept for `tiger.ipynb`.
"""

import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def plot_repair_stages(seed: int = 7, out_file: str = None):
    import matplotlib.pyplot as plt
    root = Path("data")
    
    # Load DataFrames and Report
    try:
        df_clean = pd.read_parquet(root / "sample" / "products.parquet")
        df_noisy = pd.read_parquet(root / "processed" / f"noisy_report_seed{seed}.parquet")
        df_rep = pd.read_parquet(root / "processed" / f"repaired_report_seed{seed}.parquet")
        with open(root / "outputs" / f"repair_report_seed{seed}.json") as f:
            report = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}. Make sure you run the pipeline first!")
        return

    # Find a product that was successfully repaired
    repaired_pids = [
        rid for rid, outcome in report.get("outcomes", {}).items() 
        if outcome.get("final_status") == "repaired"
    ]
    if not repaired_pids:
        print("No successfully repaired products found to visualize.")
        return
        
    # Group by error type to ensure maximum variety
    grouped = {}
    for pid in repaired_pids:
        subtype = df_noisy[df_noisy["product_id"] == pid].iloc[0].get("noise_subtype", "unknown")
        if subtype not in grouped:
            grouped[subtype] = []
        grouped[subtype].append(pid)
        
    pids = []
    
    # Priority 1: Force include the Generative Fallback image if one exists
    for pid in repaired_pids:
        log = report.get("outcomes", {}).get(pid, {}).get("log", [])
        if any(entry.get("candidate_product") == "GENERATED" for entry in log):
            pids.append(pid)
            # Remove from grouped so we don't duplicate
            for sub, lst in grouped.items():
                if pid in lst:
                    lst.remove(pid)
            break
            
    # Priority 2: Fill the rest up to 6 with diverse error types
    while len(pids) < 6 and grouped:
        for subtype in list(grouped.keys()):
            if grouped[subtype]:
                pids.append(grouped[subtype].pop(0))
                if len(pids) >= 6:
                    break
            else:
                del grouped[subtype]
                
    num_examples = len(pids)
    
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 6 * num_examples))
    # Ensure axes is 2D even if num_examples == 1
    if num_examples == 1:
        axes = [axes]
        
    for row_idx, pid in enumerate(pids):
        row_clean = df_clean[df_clean["product_id"] == pid].iloc[0]
        row_noisy = df_noisy[df_noisy["product_id"] == pid].iloc[0]
        row_rep = df_rep[df_rep["product_id"] == pid].iloc[0]
        
        error_type = row_noisy.get('noise_subtype', 'unknown')
        error_map = {
            'color_flip': 'Wrong Color (Image Swapped)',
            'near_color_flip': 'Wrong Color (Slightly Off)',
            'swap_image': 'Wrong Image Entirely',
            'swap_image_same_category': 'Wrong Image (Same Category)',
            'material_flip': 'Wrong Text (Material Altered)',
            'title_contradiction': 'Wrong Text (Contradicts Image)',
            'attribute_drop': 'Wrong Text (Missing Detail)',
            'mixed_swap_color': 'Image and Text Both Corrupted',
            'missing_image': 'Missing Image'
        }
        human_error = error_map.get(error_type, error_type)
        
        stages = [
            ("Clean (Original)", row_clean),
            (f"Corrupted\n[{human_error}]", row_noisy),
            ("Repaired", row_rep)
        ]
        
        for col_idx, (title, row) in enumerate(stages):
            ax = axes[row_idx][col_idx]
            
            # Image — resolve relative to data/ root to handle generated paths
            img_path_str = row["image_path"]
            if not img_path_str:
                img_path = None
            else:
                # Generated images are stored relative to the repo root
                candidate = Path(img_path_str)
                if not candidate.is_absolute():
                    candidate = Path("data").parent / candidate
                img_path = candidate if candidate.exists() else Path(img_path_str)

            if img_path and img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
            else:
                ax.set_facecolor("#222222")
                ax.text(0.5, 0.5, "🤖 Image Generated\nby Stable Diffusion" if col_idx == 2 and not img_path_str else "Image Missing",
                        ha="center", va="center", color="white", fontsize=10, fontweight="bold",
                        transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                
            # If it's the rightmost column (Repaired), pull the repair action from the log if possible
            if col_idx == 0:
                ax_title = "Clean (Original)"
            elif col_idx == 1:
                ax_title = f"Corrupted\n[{human_error}]"
            else:
                action = report.get("outcomes", {}).get(pid, {}).get("log", [{}])[-1].get("action", "Repaired")
                if "direction" in report.get("outcomes", {}).get(pid, {}).get("log", [{}])[-1]:
                     action = report.get("outcomes", {}).get(pid, {}).get("log", [{}])[-1]["direction"]
                ax_title = f"Repaired\n[Action: {action}]"
                
            ax.set_title(ax_title, fontsize=12, fontweight="bold")
            ax.axis("off")
            
            # Text
            text = row.get("title", "No Text")
            ax.text(0.5, -0.1, text, ha="center", va="top", transform=ax.transAxes, 
                    fontsize=11, wrap=True, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    if out_file:
        plt.savefig(out_file, bbox_inches="tight")
        print(f"Saved visualization to {out_file}")
    else:
        plt.show()

if __name__ == "__main__":
    plot_repair_stages(seed=7, out_file="repair_viz.png")


# ---------------------------------------------------------------------------
# the qualitative grid (D10)
# ---------------------------------------------------------------------------

ERROR_LABELS = {
    "color_flip": "wrong colour in text",
    "near_color_flip": "near-miss colour in text",
    "material_flip": "wrong material in text",
    "attribute_drop": "attribute missing from text",
    "title_contradiction": "title contradicts attributes",
    "swap_image": "wrong image",
    "swap_image_same_category": "wrong image, same category",
    "mixed_swap_color": "image and text both wrong",
    "missing_image": "no image",
}

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    "/Library/Fonts/Arial.ttf",
]


def _font(size: int):
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _thumb(root: Path, rel_path: str, box: int) -> Image.Image:
    """The image at `rel_path`, letterboxed into a `box`x`box` tile.

    A missing or unreadable path is a legitimate state here, not an error: an
    E4 row has no image, and a T2V row's replacement may not have been written
    if generation was disabled. Those tiles are labelled rather than skipped.
    """
    tile = Image.new("RGB", (box, box), (235, 235, 235))
    p = (root / rel_path) if rel_path else None
    if p is None or not p.exists():
        d = ImageDraw.Draw(tile)
        label = "no image" if not rel_path else "image not found"
        f = _font(12)
        w = d.textlength(label, font=f)
        d.text(((box - w) / 2, box / 2 - 6), label, fill=(120, 120, 120), font=f)
        return tile
    with Image.open(p) as im:
        im = im.convert("RGB")
        im.thumbnail((box, box))
        tile.paste(im, ((box - im.width) // 2, (box - im.height) // 2))
    return tile


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int, max_lines: int) -> str:
    """Wrap to a pixel width. Character counts are the wrong unit for a
    proportional font -- they let a caption run into the next column."""
    lines: list[str] = []
    for word in text.split():
        if lines and draw.textlength(f"{lines[-1]} {word}", font=font) <= max_w:
            lines[-1] = f"{lines[-1]} {word}"
        else:
            if len(lines) == max_lines:
                lines[-1] = lines[-1][:-1] + "\u2026"
                break
            lines.append(word)
    return "\n".join(lines)


def _caption(row: pd.Series) -> str:
    attrs = row.get("attributes", "")
    try:
        a = json.loads(attrs) if isinstance(attrs, str) else dict(attrs or {})
    except (ValueError, TypeError):
        a = {}
    parts = [f"{k}={v}" for k, v in sorted(a.items()) if v]
    return f"{str(row.get('title', '') or '(no title)')}\n" + ", ".join(parts)


def _last_action(outcome: dict) -> str:
    log = outcome.get("log") or []
    if not log:
        return outcome.get("final_status", "?")
    last = log[-1]
    if last.get("candidate_product") == "GENERATED":
        return "T2V (image synthesised)"
    if last.get("direction") == "T2V":
        return f"T2V (image from {last.get('candidate_product', '?')})"
    if last.get("direction") == "V2T":
        patch = last.get("patch") or {}
        return "V2T " + ", ".join(f"{k}->{v}" for k, v in patch.items())
    return str(last.get("action") or outcome.get("final_status", "?"))


def select_rows(report: dict, noisy: pd.DataFrame, max_rows: int = 6) -> list[str]:
    """Which repaired rows to show: generated images first, then one per error type.

    Deterministic given the report -- a figure that reshuffles between runs
    cannot be compared with the one in the paper.
    """
    outcomes = report.get("outcomes", {}) or {}
    repaired = [rid for rid, oc in outcomes.items() if oc.get("final_status") == "repaired"]
    subtype = dict(zip(noisy["row_id"].astype(str),
                       noisy.get("noise_subtype", pd.Series(dtype=str)).astype(str)))

    def generated(rid: str) -> bool:
        return any(e.get("candidate_product") == "GENERATED"
                   for e in (outcomes[rid].get("log") or []))

    chosen = [rid for rid in sorted(repaired) if generated(rid)][:1]
    seen = {subtype.get(rid, "") for rid in chosen}
    for rid in sorted(repaired):
        if len(chosen) >= max_rows:
            break
        st = subtype.get(rid, "")
        if rid not in chosen and st not in seen:
            chosen.append(rid)
            seen.add(st)
    for rid in sorted(repaired):          # top up if error types ran out
        if len(chosen) >= max_rows:
            break
        if rid not in chosen:
            chosen.append(rid)
    return chosen


def build_qualitative_grid(root: Path, seed: int = 7, out_path: Path | None = None,
                           max_rows: int = 6, tile: int = 224) -> Path | None:
    """Clean / corrupted / repaired triptychs for a handful of repaired rows.

    Reads what a run actually leaves behind -- the products table, the noisy
    frame, and the per-row repaired frame plus outcome log that
    `repair_ablation._persist_full_run` writes. Returns the path written, or
    None when the run produced no repaired row to show.
    """
    root = Path(root)
    clean = pd.read_parquet(root / "data/sample/products.parquet").set_index("row_id", drop=False)
    noisy = pd.read_parquet(root / f"data/processed/noisy_report_seed{seed}.parquet")
    rep = pd.read_parquet(root / f"data/processed/repaired_report_seed{seed}.parquet") \
        .set_index("row_id", drop=False)
    report = json.loads((root / f"data/outputs/repair_report_seed{seed}.json").read_text())
    noisy_ix = noisy.set_index("row_id", drop=False)

    rows = select_rows(report, noisy, max_rows=max_rows)
    if not rows:
        return None

    head_h, cap_h, pad = 20, 64, 10
    row_h = head_h + tile + cap_h + pad
    width = 3 * tile + 4 * pad
    canvas = Image.new("RGB", (width, len(rows) * row_h + pad), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    f_head, f_cap = _font(13), _font(11)

    for r, rid in enumerate(rows):
        oc = report["outcomes"][rid]
        st = str(noisy_ix.at[rid, "noise_subtype"]) if rid in noisy_ix.index else ""
        stages = [
            ("clean (ground truth)", clean.loc[rid] if rid in clean.index else None),
            (f"corrupted: {ERROR_LABELS.get(st, st or 'unknown')}",
             noisy_ix.loc[rid] if rid in noisy_ix.index else None),
            (f"repaired: {_last_action(oc)}", rep.loc[rid] if rid in rep.index else None),
        ]
        y = pad + r * row_h
        for c, (head, row) in enumerate(stages):
            x = pad + c * (tile + pad)
            draw.text((x, y), head[:46], fill=(20, 20, 20), font=f_head)
            if row is None:
                draw.text((x, y + head_h + 8), "(row absent)", fill=(150, 150, 150), font=f_cap)
                continue
            if isinstance(row, pd.DataFrame):     # duplicate row_id: take the first
                row = row.iloc[0]
            canvas.paste(_thumb(root, str(row.get("image_path", "") or ""), tile),
                         (x, y + head_h))
            text = _wrap(draw, _caption(row).replace("\n", " · "), f_cap, tile, 4)
            draw.multiline_text((x, y + head_h + tile + 4), text,
                                fill=(60, 60, 60), font=f_cap, spacing=2)

    out_path = Path(out_path or (root / "paper_figures/qualitative_grid_final.png"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")
    return out_path
