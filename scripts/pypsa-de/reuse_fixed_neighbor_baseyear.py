# SPDX-FileCopyrightText: 2026 Can Yalman
#
# SPDX-License-Identifier: MIT

"""Copy the common solved base-year network into a scenario result folder."""

from pathlib import Path
from shutil import copy2


source = Path(snakemake.input.reference_network)
target = Path(snakemake.output.network)
target.parent.mkdir(parents=True, exist_ok=True)
copy2(source, target)
print(f"Reused common base-year network: {source} -> {target}")
