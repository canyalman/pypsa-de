# SPDX-License-Identifier: MIT

from pathlib import Path
from shutil import copy2

source = Path(snakemake.input.source)  # noqa: F821
target = Path(snakemake.output[0])  # noqa: F821
if source.resolve() == target.resolve():
    raise ValueError("Weather cache source and target must be different.")
target.parent.mkdir(parents=True, exist_ok=True)
copy2(source, target)
print(f"Reused weather resource without modifying source: {source} -> {target}")
