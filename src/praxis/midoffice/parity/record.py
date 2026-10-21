"""The flat record: one number, fully keyed.

Every metric the engine produces is flattened to `(metric, level, entity, detail) -> value`.
The shape is chosen so that a disagreement points at exactly one number and names it, rather
than reporting that two nested structures differ somewhere.

Both sides of a comparison emit this type -- the engine and the independent reference alike --
so the comparator never knows which side it is holding.  That symmetry is what makes missing
and extra records detectable: a comparator written against one side's structure can only
report on numbers that side produced, and would call a silently dropped metric a pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

# Metric names. Flat strings rather than an enum: golden files outlive code, and a renamed
# enum member would silently fail to match an old file instead of reporting the rename.
TOTAL_RISK = "total_risk"
COMMON_FACTOR_RISK = "common_factor_risk"
SPECIFIC_RISK = "specific_risk"
BETA = "beta"
CORRELATION = "correlation"
TRACKING_ERROR = "tracking_error"

WEIGHT = "weight"
EXPOSURE = "exposure"

CTR_TOTAL = "ctr_total"
CTR_MARGINAL = "ctr_marginal"
CTR_PERCENT = "ctr_percent"
CTR_COMMON_FACTOR = "ctr_common_factor"
CTR_SPECIFIC = "ctr_specific"

ROW_EXPOSURE = "row_exposure"
ROW_STANDALONE = "row_standalone_risk"
ROW_MARGINAL = "row_marginal"
ROW_CONTRIBUTION = "row_contribution"
ROW_RISK_PROPORTION = "row_risk_proportion"
ROW_LITTERMAN_PROPORTION = "row_litterman_proportion"

VALUE_AT_RISK = "value_at_risk"
EXPECTED_SHORTFALL = "expected_shortfall"

COVERAGE_WEIGHT = "coverage_weight"
COVERAGE_COUNT = "coverage_count"

SCHEMA_VERSION = 1


@dataclass(frozen=True, order=True)
class Record:
    """One number and the four coordinates that identify it.

    `order=True` with the key fields declared before `value` means records sort by key, which
    is what makes a saved set diffable and a comparison deterministic.
    """

    metric: str
    level: str
    entity: str
    detail: str
    value: float

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.metric, self.level, self.entity, self.detail)

    def to_json(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "level": self.level,
            "entity": self.entity,
            "detail": self.detail,
            "value": self.value,
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "Record":
        return cls(
            metric=str(raw["metric"]),
            level=str(raw["level"]),
            entity=str(raw["entity"]),
            detail=str(raw["detail"]),
            value=float(raw["value"]),
        )


class DuplicateRecord(ValueError):
    """Two records share a key, so the set cannot be compared unambiguously."""


def index(records: Iterable[Record]) -> dict[tuple[str, str, str, str], float]:
    """Key every record, refusing duplicates.

    A duplicate key is an emitter bug, and left alone it would silently shadow one number
    with another -- the comparison would pass while testing only whichever arrived last.
    """
    out: dict[tuple[str, str, str, str], float] = {}
    for r in records:
        if r.key in out:
            raise DuplicateRecord(
                f"two records share key {r.key}: {out[r.key]!r} and {r.value!r}; "
                f"the emitter is producing the same number twice"
            )
        out[r.key] = r.value
    return out


def dumps(records: Iterable[Record], *, case_name: str = "", source: str = "") -> str:
    """Serialise a record set stably, with its provenance in the header.

    The header is why this is not a bare array: a golden file that does not say which case
    and which producer made it is a file nobody can act on when it disagrees.
    """
    rows = sorted(records)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "case_name": case_name,
        "source": source,
        "record_count": len(rows),
        "records": [r.to_json() for r in rows],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def save(
    records: Iterable[Record], path: str | Path, *, case_name: str = "", source: str = ""
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dumps(records, case_name=case_name, source=source), encoding="utf-8")
    return p


def load(path: str | Path) -> tuple[tuple[Record, ...], dict[str, Any]]:
    """Load a record set and its header, checking the count the file claims.

    The count is checked because a truncated file is otherwise indistinguishable from a
    smaller run, and a comparison against a truncated golden file passes on everything it
    still contains.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(raw.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
        raise ValueError(
            f"{path} declares record schema_version {raw.get('schema_version')}, "
            f"this build reads {SCHEMA_VERSION}"
        )
    records = tuple(Record.from_json(r) for r in raw["records"])
    claimed = raw.get("record_count")
    if claimed is not None and int(claimed) != len(records):
        raise ValueError(
            f"{path} claims {claimed} records but carries {len(records)}; "
            f"the file is truncated or was edited by hand"
        )
    header = {k: v for k, v in raw.items() if k != "records"}
    return records, header
