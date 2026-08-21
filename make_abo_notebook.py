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
        continue  # skip entirely

    # Fix 3: Fix the broken CSV path in the results cell
    if "repair_ablations_summary_run1.csv" in src:
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

    new_cells.append(cell)

nb["cells"] = new_cells

with open("tiger_abo.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"Fixed tiger_abo.ipynb — {len(nb['cells'])} cells total.")
print("Changes made:")
print("  1. Added 'git pull' to setup cell so latest code is always fetched")
print("  2. Removed Gemini API key cell (not needed for SigLIP-only run)")
print("  3. Fixed CSV path to dynamically find whatever file ablate-repair produces")
