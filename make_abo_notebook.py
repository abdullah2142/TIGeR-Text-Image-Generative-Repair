import json

with open("tiger_abo.ipynb", "r") as f:
    nb = json.load(f)

new_cells = []
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))

    # Fix 1: In the setup cell, add git pull and remove stale outputs
    if "git clone https://github.com/namaray" in src:
        cell["source"] = [
            "!rm -rf TIGeR-Text-Image-Generative-Repair\n",
            "!git clone https://github.com/namaray/TIGeR-Text-Image-Generative-Repair.git\n",
            "%cd TIGeR-Text-Image-Generative-Repair\n",
            "# Ensure we have the absolute latest code (including import-abo command)\n",
            "!git pull\n",
            "!pip install -e \".[dev,vlm,gen]\" -q"
        ]
        cell["outputs"] = []
        new_cells.append(cell)
        continue

    # Fix 2: Drop the Gemini key cell (not needed for ABO run)
    if "gemini api" in src or "GEMINI_API_KEY" in src or "UserSecretsClient" in src:
        continue

    # Fix 3: After calibrate, insert the missing noise step before train-arbiter
    if "tiger.cli calibrate" in src and "import-abo" not in src:
        # Add calibrate cell
        new_cells.append(cell)
        # Then insert the MISSING noise cell
        noise_md = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Inject Synthetic Noise\n",
                "This injects errors into the report split so there is something to repair. "
                "**This step is required** — without it, the Arbiter and ablation have no corrupted products to work on."
            ]
        }
        noise_code = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python -m tiger.cli noise --seed 7"
            ]
        }
        new_cells.append(noise_md)
        new_cells.append(noise_code)
        continue

    # Fix 4: Fix the broken CSV path in the results cell
    if "repair_ablations_summary_run1.csv" in src or ("glob" in src and "repair_ablations_summary" in src):
        cell["source"] = [
            "import pandas as pd\n",
            "import glob\n",
            "# Dynamically find whichever ablation CSV was produced\n",
            "csvs = sorted(glob.glob('data/outputs/repair_ablations_summary*.csv'))\n",
            "print('Found CSVs:', csvs)\n",
            "if csvs:\n",
            "    df = pd.read_csv(csvs[-1])\n",
            "    print(df.to_string())\n",
            "else:\n",
            "    print('No results CSV found. Check that ablate-repair ran successfully.')"
        ]
        cell["outputs"] = []
        new_cells.append(cell)
        continue

    # Fix 5: Renumber the markdown headers to account for the new noise step
    if cell.get("cell_type") == "markdown":
        src_list = cell.get("source", [])
        joined = "".join(src_list)
        if "## 3. Retrain Arbiter" in joined:
            cell["source"] = ["## 4. Retrain Arbiter\n", "This retrains the Logistic Regression router on the new ABO-domain noise patterns."]
        elif "## 4. Run Repair Ablation" in joined:
            cell["source"] = ["## 5. Run Repair Ablation\n", "This evaluates the repair pipeline using the Independent Verifier (SigLIP) and Generative Fallback."]
        elif "## 5. View Results" in joined:
            cell["source"] = ["## 6. View Results\n", "Compare this table with the Fashion results in your paper."]

    new_cells.append(cell)

nb["cells"] = new_cells

with open("tiger_abo.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"Fixed tiger_abo.ipynb — {len(nb['cells'])} cells total.")
print("\nFinal cell order:")
for i, c in enumerate(nb["cells"]):
    src_preview = "".join(c.get("source", []))[:60].replace("\n", " ")
    print(f"  Cell {i} ({c['cell_type'][:4]}): {src_preview}")
