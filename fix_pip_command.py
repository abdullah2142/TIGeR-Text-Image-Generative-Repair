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
            "# Uninstall old version to clear Kaggle cache without breaking Torchvision\n",
            "!pip uninstall -y tiger\n",
            "!pip install --no-cache-dir -e \".[dev,vlm,gen]\" -q\n",
            "# Confirm import-abo is registered\n",
            "!python -m tiger.cli --help | grep import"
        ]
        cell["outputs"] = []
        break

with open("tiger_abo.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Updated setup cell to avoid destroying Kaggle's PyTorch environment")
