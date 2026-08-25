import json

DIAGNOSTIC_MD = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6b. Arbiter Confidence Diagnostic\n",
        "Visualizes the Arbiter's `max(predict_proba())` distribution over all flagged ABO items.\n",
        "This reveals whether the Gamma Gate (threshold=0.60) is ever firing and whether the Arbiter\n",
        "is overconfident on out-of-domain data. See `ROADMAP_PROGRESS.md` H11 for context."
    ]
}

DIAGNOSTIC_CODE = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import json\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "from pathlib import Path\n",
        "from tiger.arbiter import ArbiterModel, featurize, CLASSES\n",
        "\n",
        "# Load trained model\n",
        "model_path = Path('data/thresholds/tiger_arbiter_model.json')\n",
        "assert model_path.exists(), 'Run train-arbiter first'\n",
        "model = ArbiterModel.from_json(model_path.read_text())\n",
        "\n",
        "# Load one evidence file from the calibration run\n",
        "ev_files = sorted(Path('data/outputs').glob('evidence_cal_seed*.jsonl'))\n",
        "assert ev_files, 'No evidence JSONL files found — run train-arbiter first'\n",
        "ev_file = ev_files[-1]  # use the last calibration seed\n",
        "\n",
        "records = [json.loads(l) for l in ev_file.read_text().splitlines() if l.strip()]\n",
        "print(f'Loaded {len(records)} evidence records from {ev_file.name}')\n",
        "\n",
        "# Compute max confidence per record and predicted class\n",
        "max_probs = []\n",
        "pred_classes = []\n",
        "for ev in records:\n",
        "    p = model.predict_proba(featurize(ev))\n",
        "    top = max(p, key=p.get)\n",
        "    max_probs.append(p[top])\n",
        "    pred_classes.append(top)\n",
        "\n",
        "max_probs = np.array(max_probs)\n",
        "GAMMA = 0.60\n",
        "\n",
        "print(f'\\nArbiter Confidence Stats (n={len(max_probs)})')\n",
        "print(f'  Mean max-p  : {max_probs.mean():.3f}')\n",
        "print(f'  Median max-p: {np.median(max_probs):.3f}')\n",
        "print(f'  Min max-p   : {max_probs.min():.3f}')\n",
        "print(f'  % below gamma={GAMMA}: {(max_probs < GAMMA).mean()*100:.1f}%')\n",
        "print(f'  Predicted class distribution: { {c: pred_classes.count(c) for c in CLASSES} }')\n",
        "\n",
        "# Plot\n",
        "fig, ax = plt.subplots(figsize=(9, 4))\n",
        "ax.hist(max_probs, bins=30, color='steelblue', edgecolor='white', alpha=0.85)\n",
        "ax.axvline(GAMMA, color='red', linestyle='--', linewidth=1.5, label=f'Gamma threshold ({GAMMA})')\n",
        "ax.set_xlabel('Max Arbiter Confidence (max p over classes)', fontsize=12)\n",
        "ax.set_ylabel('Count', fontsize=12)\n",
        "ax.set_title('ABO Arbiter Confidence Distribution', fontsize=12)\n",
        "ax.legend()\n",
        "plt.tight_layout()\n",
        "plt.savefig('data/outputs/arbiter_confidence_abo.png', dpi=150)\n",
        "plt.show()\n",
        "print('Saved: data/outputs/arbiter_confidence_abo.png')"
    ]
}

with open("tiger_abo.ipynb", "r") as f:
    nb = json.load(f)

# Insert AFTER the ablate-repair cell (cell 12 by current numbering),
# which is also right before the final results cell
insert_after_text = "ablate-repair"
insert_idx = None
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "ablate-repair" in src and cell.get("cell_type") == "code":
        insert_idx = i + 1  # insert after this cell
        break

if insert_idx is not None:
    nb["cells"].insert(insert_idx, DIAGNOSTIC_CODE)
    nb["cells"].insert(insert_idx, DIAGNOSTIC_MD)
    print(f"Inserted diagnostic cells after cell {insert_idx - 1} (ablate-repair)")
else:
    print("ERROR: Could not find ablate-repair cell to insert after")

with open("tiger_abo.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

# Print final cell order
print("\nFinal cell order:")
for i, c in enumerate(nb["cells"]):
    preview = "".join(c.get("source", []))[:60].replace("\n", " ").strip()
    print(f"  Cell {i} ({c['cell_type'][:4]}): {preview}")
