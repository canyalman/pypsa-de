# SPDX-FileCopyrightText: 2026 Can Yalman
#
# SPDX-License-Identifier: MIT

"""Copy a common solved network into a scenario result folder."""

from pathlib import Path
from shutil import copy2

source = Path(snakemake.input.reference_network)  # noqa: F821
target = Path(snakemake.output.network)  # noqa: F821
target.parent.mkdir(parents=True, exist_ok=True)
copy2(source, target)
print(f"Reused common reference network: {source} -> {target}")
