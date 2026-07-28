import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def plot_repair_stages(seed: int = 7, out_file: str = None):
    root = Path("data")
    
    import json
    
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
            
            # Image
            img_path = Path(row["image_path"])
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Image Missing", ha="center", va="center")
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
