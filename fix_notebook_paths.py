import json

with open("tiger_abo.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "tiger.cli import-abo" in src:
        cell["source"] = [
            "!python -m tiger.cli import-abo \\\n",
            "    --listings-dir /kaggle/input/amazon-berkeley-objects-small/abo-listings/listings/metadata \\\n",
            "    --images-csv /kaggle/input/amazon-berkeley-objects-small/abo-images-small/images/metadata/images.csv \\\n",
            "    --images-dir /kaggle/input/amazon-berkeley-objects-small/abo-images-small/images/small"
        ]

with open("tiger_abo.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Updated notebook with exact paths from Kaggle screenshot")
