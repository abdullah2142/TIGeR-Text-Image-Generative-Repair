import json

with open("tiger_abo.ipynb", "r") as f:
    nb = json.load(f)

new_cells = []
for cell in nb["cells"]:
    new_cells.append(cell)
    # Check if this cell is the import-abo command
    src = "".join(cell.get("source", []))
    if "tiger.cli import-abo" in src:
        # Add a validation cell immediately after it
        validation_md = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Validate Data Import\n",
                "This ensures the notebook stops immediately if the ABO data wasn't found or parsed correctly."
            ]
        }
        validation_code = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "from pathlib import Path\n",
                "\n",
                "parquet_file = Path('data/sample/products.parquet')\n",
                "assert parquet_file.exists(), \"Data extraction failed: products.parquet not found! Check your --listings-dir and --images-dir paths.\"\n",
                "\n",
                "df = pd.read_parquet(parquet_file)\n",
                "print(f\"Successfully imported {len(df)} products.\")\n",
                "assert len(df) > 0, \"Data extraction failed: Parquet file is empty!\"\n"
            ]
        }
        # Only append if we haven't already (prevent duplicates if run multiple times)
        if not any("Validate Data Import" in "".join(c.get("source", [])) for c in nb["cells"]):
            new_cells.append(validation_md)
            new_cells.append(validation_code)

nb["cells"] = new_cells

with open("tiger_abo.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Added validation cell to notebook.")
