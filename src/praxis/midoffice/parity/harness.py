"""The entry point: run one case through both sides and report.

`KNOWN_GAPS` is the declared list of numbers the reference computes and the engine does not
expose.  It is deliberately a hand-maintained list with a reason against each entry rather
than a tolerance or a filter, for two reasons: an undeclared gap fails, so a metric that
quietly disappears is caught; and a declared gap that is no longer a gap also fails, so the
list cannot accumulate stale excuses.  Every entry here is a small, specific statement about
the engine's reporting surface, and the right resolution for each is to close the gap and
delete the line.
"""

from __future__ import annotations

from .case import ParityCase
from .compare import Report, compare, policy_for
from .engine import engine_records
from .record import COVERAGE_WEIGHT
from .reference import reference_records

KNOWN_GAPS: dict[tuple[str, str, str, str], str] = {
    (COVERAGE_WEIGHT, "benchmark", "", "uncovered"): (
        "CoverageSummary reports uncovered weight for the portfolio only. The benchmark's "
        "uncovered weight matters for the same reason the portfolio's does -- it scales "
        "benchmark risk and therefore beta and tracking error -- so this is a reporting gap "
        "to close, not a number that does not exist."
    ),
}


def run_parity(case: ParityCase) -> Report:
    """Compute both sides and compare under the tolerances the case's conventions imply."""
    return compare(
        reference_records(case),
        engine_records(case),
        policy=policy_for(case),
        known_gaps=KNOWN_GAPS,
        case_name=case.name,
    )
