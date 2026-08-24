import json

RECAL_MD = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6c. ABO Gamma Recalibration\n",
        "The diagnostic above shows the Arbiter is overconfident on ABO — confidence scores are\n",
        "clustered above the default gamma=0.60, so the gate never fires.\n",
        "This cell computes a domain-specific gamma at the 25th percentile of the observed\n",
        "confidence distribution, patches the config, and reruns the ablation.\n",
        "The new table should show Full System > No Gamma Gate."
    ]
}

RECAL_CODE = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import json, yaml, subprocess\n",
        "import numpy as np\n",
        "from pathlib import Path\n",
        "from tiger.arbiter import ArbiterModel, featurize\n",
        "\n",
        "# 1. Load model and compute confidence distribution\n",
        "model = ArbiterModel.from_json(Path('data/thresholds/tiger_arbiter_model.json').read_text())\n",
        "ev_files = sorted(Path('data/outputs').glob('evidence_cal_seed*.jsonl'))\n",
        "records = [json.loads(l) for l in ev_files[-1].read_text().splitlines() if l.strip()]\n",
        "max_probs = np.array([max(model.predict_proba(featurize(ev)).values()) for ev in records])\n",
        "\n",
        "# 2. Set new gamma at 25th percentile (bottom 25% escalates)\n",
        "new_gamma = float(np.percentile(max_probs, 25))\n",
        "print(f'Default gamma : 0.60')\n",
        "print(f'ABO gamma (p25): {new_gamma:.3f}')\n",
        "print(f'Items that would now escalate via gamma: {(max_probs < new_gamma).sum()} / {len(max_probs)}')\n",
        "\n",
        "# 3. Patch configs/tiger.yaml with new gamma\n",
        "cfg_path = Path('configs/tiger.yaml')\n",
        "cfg = yaml.safe_load(cfg_path.read_text())\n",
        "if 'arbiter' not in cfg:\n",
        "    cfg['arbiter'] = {}\n",
        "cfg['arbiter']['gamma'] = round(new_gamma, 3)\n",
        "cfg_path.write_text(yaml.dump(cfg, default_flow_style=False))\n",
        "print(f'\\nPatched configs/tiger.yaml: arbiter.gamma = {new_gamma:.3f}')\n",
        "\n",
        "# 4. Rerun ablation with new gamma\n",
        "print('\\nRunning ablate-repair with recalibrated gamma...')\n",
        "result = subprocess.run(\n",
        "    ['python', '-m', 'tiger.cli', 'ablate-repair', '--independent', '--generative-fallback'],\n",
        "    capture_output=True, text=True\n",
        ")\n",
        "print(result.stdout[-2000:] if result.stdout else '')\n",
        "if result.returncode != 0:\n",
        "    print('STDERR:', result.stderr[-1000:])\n",
        "\n",
        "# 5. Show before/after comparison\n",
        "import pandas as pd, glob\n",
        "csvs = sorted(glob.glob('data/outputs/repair_ablations_summary*.csv'))\n",
        "if csvs:\n",
        "    df = pd.read_csv(csvs[-1])\n",
        "    print('\\n=== Ablation Results with Recalibrated Gamma ===')\n",
        "    print(df.to_string())\n",
        "    print('\\nIf Full System > No Gamma Gate (gamma=0), gamma recalibration is working.')\n"
    ]
}

with open("tiger_abo.ipynb", "r") as f:
    nb = json.load(f)

# Insert after the diagnostic cell (which is after ablate-repair)
# Find the diagnostic markdown cell
insert_idx = None
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "Arbiter Confidence Diagnostic" in src and cell.get("cell_type") == "markdown":
        # Insert after both the markdown and code cells for the diagnostic (i and i+1)
        insert_idx = i + 2
        break

if insert_idx is not None:
    nb["cells"].insert(insert_idx, RECAL_CODE)
    nb["cells"].insert(insert_idx, RECAL_MD)
    print(f"Inserted recalibration cells at position {insert_idx}")
else:
    print("ERROR: Could not find diagnostic cell")

with open("tiger_abo.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("\nFinal cell order:")
for i, c in enumerate(nb["cells"]):
    preview = "".join(c.get("source", []))[:55].replace("\n", " ").strip()
    print(f"  Cell {i} ({c['cell_type'][:4]}): {preview}")
