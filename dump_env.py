# dump_env.py
# Prints environment + repo info + script --help outputs (Windows / VS Code friendly)

import os
import sys
import subprocess
from pathlib import Path

ROOT = Path.cwd()

def hr(title: str) -> None:
    print("\n" + "=" * 12 + title + "=" * 12)

def run(cmd, shell=False):
    """Run a command and print it + output (safe for VS Code terminal)."""
    print(f"\n$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    try:
        out = subprocess.check_output(cmd, shell=shell, stderr=subprocess.STDOUT, text=True)
        print(out.rstrip())
    except subprocess.CalledProcessError as e:
        print(e.output.rstrip())
    except FileNotFoundError as e:
        print(f"(command not found) {e}")

def list_dir(p: Path):
    if not p.exists():
        print(f"(missing) {p}")
        return
    for item in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
        print(item.name)

def main():
    hr("PWD")
    print(str(ROOT))

    hr("PYTHON")
    print("python version:", sys.version.replace("\n", " "))
    print("executable:", sys.executable)

    # where.exe python (Windows)
    run(["where", "python"])

    hr("PIP")
    run([sys.executable, "-m", "pip", "--version"])

    hr("PROJECT FILES (top-level)")
    list_dir(ROOT)

    hr("SCRIPTS (*.py top-level)")
    for f in sorted(ROOT.glob("*.py")):
        print(f.name)

    hr("DATA FOLDERS")
    data = ROOT / "data"
    processed = ROOT / "data" / "processed"
    print("data/:")
    list_dir(data)
    print("\ndata/processed/:")
    list_dir(processed)

    hr("PARQUETS FOUND (recursive)")
    parqs = sorted(ROOT.rglob("*.parquet"))
    if parqs:
        for p in parqs:
            print(str(p))
    else:
        print("(no .parquet found)")

    hr("RESULTS FOLDERS")
    results = ROOT / "results"
    list_dir(results)

    # Script help outputs (only run if file exists)
    hr("SCRIPT HELPS")
    scripts = ["run_sieve.py", "evaluate_pipeline.py", "eval_locked_transitions.py", "mismatch_analyzer.py", "upgrade6_vlm_repair.py"]
    for s in scripts:
        sp = ROOT / s
        if sp.exists():
            run([sys.executable, str(sp), "--help"])
        else:
            print(f"\n$ {s} --help\n(missing {s})")

if __name__ == "__main__":
    main()
