# SPDX-License-Identifier: MIT

"""Cache selection respects native multi-output jobs, not just file existence."""

import logging
import re

from snakemake.exceptions import WildcardError
from snakemake.io import apply_wildcards, regex_from_filepattern

from scripts.controlled_weather import cached_weather_resources

logger = logging.getLogger(__name__)


def reusable_weather_resources(directory, rules, resource_prefix):
    """
    Never reuse a partial output group that still needs its native producer.

    An individual copy job is safe only when *all* outputs of every overlapping
    native job are cacheable and present. Iterate to a fixed point because one
    producer can share an output with another multi-output producer.
    """
    candidates = set(cached_weather_resources(directory))
    prefix = str(resource_prefix).replace("\\", "/")
    groups = set()
    matched = set()
    for rule in rules:
        patterns = [str(output).replace("\\", "/") for output in rule.output]
        for pattern in patterns:
            if not pattern.startswith(prefix):
                continue
            expression = re.compile(regex_from_filepattern(pattern[len(prefix) :]))
            for candidate in candidates:
                match = expression.fullmatch(candidate)
                if match is None:
                    continue
                required = set()
                local_group = True
                for output in patterns:
                    if not output.startswith(prefix):
                        local_group = False
                        continue
                    try:
                        required.add(
                            apply_wildcards(output[len(prefix) :], match.groupdict())
                        )
                    except WildcardError:
                        local_group = False
                required.add(candidate)
                groups.add((frozenset(required), local_group))
                matched.add(candidate)
    allowed = candidates.intersection(matched)
    while True:
        previous = allowed.copy()
        for required, local_group in groups:
            if not local_group or not required.issubset(allowed):
                allowed.difference_update(required)
        if allowed == previous:
            break
    logger.info(
        "Weather cache: %s reusable files; %s excluded by complete-output-group checks.",
        len(allowed),
        len(candidates - allowed),
    )
    return sorted(allowed)
