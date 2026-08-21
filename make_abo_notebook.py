import json

with open("tiger_abo.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "git clone https://github.com/namaray" in src:
        cell["source"] = [
            "!rm -rf TIGeR-Text-Image-Generative-Repair\n",
            "!git clone https://github.com/namaray/TIGeR-Text-Image-Generative-Repair.git\n",
            "%cd TIGeR-Text-Image-Generative-Repair\n",
            "!git pull\n",
            "# Force reinstall to bypass Kaggle's cached old package\n",
            "!pip install --force-reinstall -e \".[dev,vlm,gen]\" -q\n",
            "# Confirm import-abo is registered\n",
            "!python -m tiger.cli --help | grep import"
        ]
        cell["outputs"] = []
        break

with open("tiger_abo.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Updated setup cell with --force-reinstall")
