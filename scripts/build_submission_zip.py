#!/usr/bin/env python3
"""Build submission zip bundling lerobot source and model_weights."""

import os
import sys
import shutil
import zipfile
from pathlib import Path

def build_zip():
    repo_root = Path(__file__).resolve().parent.parent
    sub_template = repo_root / "submission_template"
    output_zip = repo_root / "submissions" / "submission_EXP_001.zip"
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building submission zip: {output_zip}")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(sub_template):
            if "__pycache__" in root:
                continue
            for f in files:
                if f.endswith(".pyc"):
                    continue
                full_path = Path(root) / f
                rel_path = full_path.relative_to(sub_template)
                z.write(full_path, rel_path)

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"Successfully built {output_zip} (Size: {size_mb:.2f} MB)")

if __name__ == "__main__":
    build_zip()
