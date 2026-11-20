"""The model store: a risk model's forecasts by date, in Parquet, each looked up in milliseconds.

midoffice20. A build writes one model's forecast as of every date it covers -- the factor
covariance as the risk engine uses it, the exposures, the specific risk -- with its factor
returns, under the model's own `model_id`, so a variant is one more build beside it. The keys
are the lookup columns Core's DataTables will read when the risk GUI reads the store:
(model_id, covariance_date) for the covariance and the specific risk, (model_id,
exposure_date) for the exposures, model_id for the factor returns. Nothing here uses Core.

Under the store's root, one directory a model:

    <model_id>/manifest.json             what the model is, how it was built, the build's checks
    <model_id>/factor_covariance.parquet model_id, covariance_date, factor, a column a factor
    <model_id>/exposures.parquet         model_id, exposure_date, security, security_key,
                                         ticker, a column a factor
    <model_id>/specific_risk.parquet     model_id, covariance_date, security, specific_risk
    <model_id>/factor_returns.parquet    model_id, date, a column a factor
    <model_id>/multipliers.parquet       model_id, covariance_date, the regime multipliers its
                                         covariance carries
    <model_id>/bias_reference.parquet    model_id, covariance_date, portfolio, forecast: a stored
                                         bias run's own forecasts, where the build was given one

Every number is as the engine reads it: the covariance annualised, through
`risk.basis.nearest_psd` and times its regime multipliers (`bias.RiskForecast`, as the bias
statistics tested it); the specific risk an annualised volatility; the factor returns daily, as
estimated. A dated table holds one row group a date, so a date's lookup reads that row group
alone, however many dates the store holds. A build writes into `.partial/<model_id>`, where its
checks read the finished files, and renames it into place when they pass (`commit`), so a model
directory is never half written or unchecked; it refuses to replace one without `overwrite`.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STORE_ROOT = Path("data/model_store")
"""The store's default root, under `data/` (gitignored): the build's output, not the repo's."""

COVARIANCE = "factor_covariance"
EXPOSURES = "exposures"
SPECIFIC = "specific_risk"
FACTOR_RETURNS = "factor_returns"
MULTIPLIERS = "multipliers"
BIAS_REFERENCE = "bias_reference"
MANIFEST = "manifest.json"
PARTIAL = ".partial"

_COMPRESSION = "zstd"
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _schema(fixed: Sequence[tuple[str, pa.DataType]], factors: Sequence[str]) -> pa.Schema:
    names = [name for name, _ in fixed]
    clash = sorted(set(names) & set(factors))
    if clash:
        raise ValueError(f"a factor may not be named as a key column: {clash}")
    return pa.schema(list(fixed) + [(name, pa.float64()) for name in factors])


class ModelWriter:
    """Writes one model's store, a date at a time, in date order.

    `factor_names` fixes the factor order of every table. `add` takes one date's forecast;
    `finish` writes the factor returns and the manifest into `.partial/<model_id>`, which
    `StoredModel(writer.partial.parent, model_id)` reads; `commit` moves it into place."""

    def __init__(self, root: Path | str, model_id: str, factor_names: Sequence[str], *,
                 overwrite: bool = False) -> None:
        if not _MODEL_ID.fullmatch(model_id or ""):
            raise ValueError(f"a model id is letters, digits, '-' and '_', a directory's name: "
                             f"{model_id!r}")
        self.root = Path(root)
        self.model_id = model_id
        self.factor_names = tuple(factor_names)
        if len(set(self.factor_names)) != len(self.factor_names):
            raise ValueError("the factor names repeat")
        self.final = self.root / model_id
        if self.final.exists() and not overwrite:
            raise FileExistsError(f"{self.final} exists; the store refuses to replace a model "
                                  f"without overwrite")
        self.partial = self.root / PARTIAL / model_id
        if self.partial.exists():
            shutil.rmtree(self.partial)
        self.partial.mkdir(parents=True)
        self.overwrite = overwrite
        factors = self.factor_names
        self._schemas = {
            COVARIANCE: _schema([("model_id", pa.string()), ("covariance_date", pa.date32()),
                                 ("factor", pa.string())], factors),
            EXPOSURES: _schema([("model_id", pa.string()), ("exposure_date", pa.date32()),
                                ("security", pa.string()), ("security_key", pa.string()),
                                ("ticker", pa.string())], factors),
            SPECIFIC: pa.schema([("model_id", pa.string()), ("covariance_date", pa.date32()),
                                 ("security", pa.string()), ("specific_risk", pa.float64())]),
        }
        self._writers = {
            name: pq.ParquetWriter(self.partial / f"{name}.parquet", schema,
                                   compression=_COMPRESSION)
            for name, schema in self._schemas.items()
        }
        self._multipliers: list[dict] = []
        self._multiplier_names: tuple[str, ...] | None = None
        self.dates: list[date] = []

    def add(self, day: date, covariance: np.ndarray, securities: Sequence[str],
            keys: Sequence[str], tickers: Sequence[str], exposures: np.ndarray,
            specific: np.ndarray, multipliers: Mapping[str, float]) -> None:
        """One date's forecast: the covariance (K x K), and the securities with a forecast, each
        with its key, its ticker that day, its exposures (N x K) and its specific risk (N)."""
        if self.dates and day <= self.dates[-1]:
            raise ValueError(f"the store is written in date order: {day} after {self.dates[-1]}")
        k = len(self.factor_names)
        covariance = np.asarray(covariance, dtype=float)
        exposures = np.asarray(exposures, dtype=float)
        specific = np.asarray(specific, dtype=float)
        n = len(securities)
        if covariance.shape != (k, k):
            raise ValueError(f"{day}: a covariance of shape {covariance.shape} for {k} factors")
        if exposures.shape != (n, k) or specific.shape != (n,) or not (
                len(keys) == len(tickers) == n):
            raise ValueError(f"{day}: {n} securities, exposures {exposures.shape}, specific "
                             f"{specific.shape}, {len(keys)} keys, {len(tickers)} tickers")
        if not (np.isfinite(covariance).all() and np.isfinite(exposures).all()
                and np.isfinite(specific).all()):
            raise ValueError(f"{day}: a forecast in the store is finite throughout")
        names = tuple(multipliers)
        if self._multiplier_names is None:
            self._multiplier_names = names
        elif names != self._multiplier_names:
            raise ValueError(f"{day}: multipliers {names}, not {self._multiplier_names}")
        model = [self.model_id]
        self._write(COVARIANCE, {
            "model_id": model * k, "covariance_date": [day] * k,
            "factor": list(self.factor_names),
            **{name: covariance[:, j] for j, name in enumerate(self.factor_names)}})
        self._write(EXPOSURES, {
            "model_id": model * n, "exposure_date": [day] * n, "security": list(securities),
            "security_key": list(keys), "ticker": list(tickers),
            **{name: exposures[:, j] for j, name in enumerate(self.factor_names)}})
        self._write(SPECIFIC, {
            "model_id": model * n, "covariance_date": [day] * n, "security": list(securities),
            "specific_risk": specific})
        self._multipliers.append({"covariance_date": day,
                                  **{name: float(v) for name, v in multipliers.items()}})
        self.dates.append(day)

    def _write(self, name: str, columns: dict) -> None:
        # One call, one row group: the date's.
        self._writers[name].write_table(pa.table(columns, schema=self._schemas[name]))

    def finish(self, factor_return_dates: Sequence[date], factor_returns: np.ndarray,
               manifest: Mapping) -> Path:
        """Write the factor returns (dates x factors, daily) and the manifest. Returns the
        partial directory, which the checks read before `commit`."""
        for writer in self._writers.values():
            writer.close()
        factor_returns = np.asarray(factor_returns, dtype=float)
        if factor_returns.shape != (len(factor_return_dates), len(self.factor_names)):
            raise ValueError(f"factor returns of shape {factor_returns.shape} for "
                             f"{len(factor_return_dates)} dates and {len(self.factor_names)} "
                             f"factors")
        schema = _schema([("model_id", pa.string()), ("date", pa.date32())], self.factor_names)
        pq.write_table(pa.table({
            "model_id": [self.model_id] * len(factor_return_dates),
            "date": list(factor_return_dates),
            **{name: factor_returns[:, j] for j, name in enumerate(self.factor_names)}},
            schema=schema), self.partial / f"{FACTOR_RETURNS}.parquet", compression=_COMPRESSION)
        names = self._multiplier_names or ()
        pq.write_table(pa.table({
            "model_id": [self.model_id] * len(self._multipliers),
            "covariance_date": [row["covariance_date"] for row in self._multipliers],
            **{name: [row[name] for row in self._multipliers] for name in names}},
            schema=_schema([("model_id", pa.string()), ("covariance_date", pa.date32())],
                           names)),
            self.partial / f"{MULTIPLIERS}.parquet", compression=_COMPRESSION)
        record = {"model_id": self.model_id, "factor_names": list(self.factor_names),
                  "dates": {"first": str(self.dates[0]), "last": str(self.dates[-1]),
                            "count": len(self.dates)},
                  **dict(manifest)}
        _write_manifest(self.partial / MANIFEST, record)
        return self.partial

    def reference(self, rows: Sequence[tuple[date, str, float]]) -> None:
        """A stored bias run's own forecasts, (date, portfolio, forecast), kept beside the model
        so a report can check itself against them."""
        pq.write_table(pa.table({
            "model_id": [self.model_id] * len(rows),
            "covariance_date": [r[0] for r in rows], "portfolio": [r[1] for r in rows],
            "forecast": [float(r[2]) for r in rows]},
            schema=pa.schema([("model_id", pa.string()), ("covariance_date", pa.date32()),
                              ("portfolio", pa.string()), ("forecast", pa.float64())])),
            self.partial / f"{BIAS_REFERENCE}.parquet", compression=_COMPRESSION)

    def commit(self, manifest_update: Mapping | None = None) -> Path:
        """Merge `manifest_update` -- the checks -- into the manifest and move the model into
        place, replacing one there only under `overwrite`. Returns its directory."""
        path = self.partial / MANIFEST
        record = json.loads(path.read_text(encoding="utf-8"))
        record.update(dict(manifest_update or {}))
        _write_manifest(path, record)
        if self.final.exists():
            if not self.overwrite:
                raise FileExistsError(f"{self.final} exists; the store refuses to replace a "
                                      f"model without overwrite")
            shutil.rmtree(self.final)
        self.partial.rename(self.final)
        return self.final


def _write_manifest(path: Path, record: Mapping) -> None:
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8",
                    newline="\n")


class _Dated:
    """A dated table's row groups, one a date, indexed by date from the file's statistics."""

    def __init__(self, path: Path, date_column: str, model_id: str | None) -> None:
        self.path = path
        self.file = pq.ParquetFile(path)
        meta = self.file.metadata
        column = self.file.schema_arrow.get_field_index(date_column)
        self.index: dict[date, int] = {}
        for i in range(meta.num_row_groups):
            stats = meta.row_group(i).column(column).statistics
            if stats is None or not stats.has_min_max or stats.min != stats.max:
                raise ValueError(f"{path}: row group {i} is not one date's")
            if stats.min in self.index:
                raise ValueError(f"{path}: {stats.min} in two row groups")
            self.index[stats.min] = i
        if model_id is not None and meta.num_row_groups:
            first = self.file.read_row_group(0, columns=["model_id"]).column(0)
            if first[0].as_py() != model_id:
                raise ValueError(f"{path} holds {first[0].as_py()!r}, not {model_id!r}")

    def read(self, day: date) -> pa.Table:
        try:
            group = self.index[day]
        except KeyError:
            raise KeyError(f"{self.path.name} has no row for {day}") from None
        return self.file.read_row_group(group)

    def close(self) -> None:
        self.file.close()


@dataclass(frozen=True)
class StoredForecast:
    """One date's forecast from the store, in arrays, in the store's factor order."""

    model_id: str
    date: date
    factor_names: tuple[str, ...]
    covariance: np.ndarray
    securities: tuple[str, ...]
    security_keys: tuple[str, ...]
    tickers: tuple[str, ...]
    exposures: np.ndarray
    specific_risk: np.ndarray

    def exposure_of(self, factor: str) -> np.ndarray:
        return self.exposures[:, self.factor_names.index(factor)]


class StoredModel:
    """One model in the store, opened once and read by date."""

    def __init__(self, root: Path | str, model_id: str) -> None:
        self.root = Path(root)
        self.model_id = model_id
        self.directory = self.root / model_id
        if not self.directory.is_dir():
            available = ", ".join(list_models(self.root)) or "none"
            raise KeyError(f"no model {model_id!r} in {self.root} (it holds: {available})")
        self.manifest = json.loads((self.directory / MANIFEST).read_text(encoding="utf-8"))
        self.factor_names = tuple(self.manifest["factor_names"])
        self._covariance = _Dated(self.directory / f"{COVARIANCE}.parquet", "covariance_date",
                                  model_id)
        self._exposures = _Dated(self.directory / f"{EXPOSURES}.parquet", "exposure_date",
                                 model_id)
        self._specific = _Dated(self.directory / f"{SPECIFIC}.parquet", "covariance_date",
                                model_id)
        self.dates = tuple(sorted(self._covariance.index))
        self.exposure_dates = tuple(sorted(self._exposures.index))
        self._factor_returns: tuple[tuple[date, ...], np.ndarray] | None = None
        self._multipliers: dict[date, dict[str, float]] | None = None

    def close(self) -> None:
        """Release the open files: Windows will not move a directory whose files are open."""
        for table in (self._covariance, self._exposures, self._specific):
            table.close()

    def __enter__(self) -> "StoredModel":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def covariance(self, day: date) -> np.ndarray:
        table = self._covariance.read(day)
        factors = tuple(table.column("factor").to_pylist())
        if factors != self.factor_names:
            raise ValueError(f"{day}: the covariance's rows are not the store's factors")
        return np.column_stack([table.column(name).to_numpy() for name in self.factor_names])

    def exposures(self, day: date) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...],
                                            np.ndarray]:
        """The securities with a forecast on `day`, their keys and tickers, and their exposures
        (securities x factors)."""
        table = self._exposures.read(day)
        matrix = np.column_stack([table.column(name).to_numpy() for name in self.factor_names])
        return (tuple(table.column("security").to_pylist()),
                tuple(table.column("security_key").to_pylist()),
                tuple(table.column("ticker").to_pylist()), matrix)

    def specific_risk(self, day: date) -> tuple[tuple[str, ...], np.ndarray]:
        table = self._specific.read(day)
        return (tuple(table.column("security").to_pylist()),
                table.column("specific_risk").to_numpy())

    def forecast(self, day: date) -> StoredForecast:
        """The date's whole forecast: the covariance, and the exposures and specific risk of the
        securities with one, in one order."""
        securities, keys, tickers, exposures = self.exposures(day)
        specific_securities, specific = self.specific_risk(day)
        if specific_securities != securities:
            raise ValueError(f"{day}: the exposures and the specific risk hold other securities")
        return StoredForecast(self.model_id, day, self.factor_names, self.covariance(day),
                              securities, keys, tickers, exposures, specific)

    def factor_returns(self, through: date | None = None) -> tuple[tuple[date, ...], np.ndarray]:
        """The daily factor returns as estimated, dates x factors, through `through` if given."""
        if self._factor_returns is None:
            table = pq.read_table(self.directory / f"{FACTOR_RETURNS}.parquet")
            dates = tuple(table.column("date").to_pylist())
            matrix = np.column_stack([table.column(name).to_numpy()
                                      for name in self.factor_names])
            self._factor_returns = (dates, matrix)
        dates, matrix = self._factor_returns
        if through is None:
            return dates, matrix
        k = int(np.searchsorted(np.array(dates, dtype="datetime64[D]"),
                                np.datetime64(through, "D"), side="right"))
        return dates[:k], matrix[:k]

    def bias_reference(self) -> dict[date, dict[str, float]]:
        """The stored bias run's forecasts kept beside the model, by date and portfolio; empty
        if the build was given none."""
        path = self.directory / f"{BIAS_REFERENCE}.parquet"
        if not path.exists():
            return {}
        table = pq.read_table(path)
        out: dict[date, dict[str, float]] = {}
        for day, name, value in zip(table.column("covariance_date").to_pylist(),
                                    table.column("portfolio").to_pylist(),
                                    table.column("forecast").to_pylist()):
            out.setdefault(day, {})[name] = value
        return out

    def multipliers(self, day: date) -> dict[str, float]:
        if self._multipliers is None:
            table = pq.read_table(self.directory / f"{MULTIPLIERS}.parquet")
            names = [n for n in table.column_names if n not in ("model_id", "covariance_date")]
            days = table.column("covariance_date").to_pylist()
            self._multipliers = {
                d: {name: table.column(name)[i].as_py() for name in names}
                for i, d in enumerate(days)}
        return dict(self._multipliers[day])


PORTFOLIOS = "portfolios"
"""Sample books beside the models: `<root>/portfolios/`, holdings by date, model-independent --
a book's holdings are data, which any model in the store can measure."""

HOLDINGS = "holdings"
PORTFOLIO_DATES = "dates"

_HOLDINGS_SCHEMA = pa.schema([
    ("portfolio", pa.string()), ("holdings_date", pa.date32()), ("security", pa.string()),
    ("security_key", pa.string()), ("ticker", pa.string()), ("quantity", pa.float64()),
    ("price", pa.float64()), ("market_value", pa.float64()), ("weight", pa.float64())])


class HoldingsWriter:
    """Writes the sample books' holdings, one row group a date holding every book's rows, and a
    table of each date's facts; into `.partial/portfolios`, moved into place by `commit`."""

    def __init__(self, root: Path | str, *, overwrite: bool = False) -> None:
        self.root = Path(root)
        self.final = self.root / PORTFOLIOS
        if self.final.exists() and not overwrite:
            raise FileExistsError(f"{self.final} exists; the store refuses to replace the "
                                  f"portfolios without overwrite")
        self.overwrite = overwrite
        self.partial = self.root / PARTIAL / PORTFOLIOS
        if self.partial.exists():
            shutil.rmtree(self.partial)
        self.partial.mkdir(parents=True)
        self._writer = pq.ParquetWriter(self.partial / f"{HOLDINGS}.parquet", _HOLDINGS_SCHEMA,
                                        compression=_COMPRESSION)
        self._facts: list[dict] = []
        self.dates: list[date] = []

    def add(self, day: date, rows: Sequence[tuple], facts: Mapping[str, object]) -> None:
        """One date: rows of (portfolio, security, security_key, ticker, quantity, price,
        market_value, weight), and the date's facts (a flat mapping, the same keys each date)."""
        if self.dates and day <= self.dates[-1]:
            raise ValueError(f"written in date order: {day} after {self.dates[-1]}")
        columns = list(zip(*rows)) if rows else [[]] * 8
        self._writer.write_table(pa.table({
            "portfolio": list(columns[0]), "holdings_date": [day] * len(rows),
            "security": list(columns[1]), "security_key": list(columns[2]),
            "ticker": list(columns[3]), "quantity": list(columns[4]), "price": list(columns[5]),
            "market_value": list(columns[6]), "weight": list(columns[7])},
            schema=_HOLDINGS_SCHEMA))
        self._facts.append({"holdings_date": day, **dict(facts)})
        self.dates.append(day)

    def finish(self, manifest: Mapping) -> Path:
        self._writer.close()
        pq.write_table(pa.Table.from_pylist(self._facts), self.partial / f"{PORTFOLIO_DATES}"
                       f".parquet", compression=_COMPRESSION)
        _write_manifest(self.partial / MANIFEST, dict(manifest))
        return self.partial

    def commit(self, manifest_update: Mapping | None = None) -> Path:
        path = self.partial / MANIFEST
        record = json.loads(path.read_text(encoding="utf-8"))
        record.update(dict(manifest_update or {}))
        _write_manifest(path, record)
        if self.final.exists():
            if not self.overwrite:
                raise FileExistsError(f"{self.final} exists; refusing without overwrite")
            shutil.rmtree(self.final)
        self.partial.rename(self.final)
        return self.final


@dataclass(frozen=True)
class Holdings:
    """One book on one date, in arrays."""

    portfolio: str
    date: date
    securities: tuple[str, ...]
    security_keys: tuple[str, ...]
    tickers: tuple[str, ...]
    quantity: np.ndarray
    price: np.ndarray
    market_value: np.ndarray
    weight: np.ndarray

    @property
    def value(self) -> float:
        return float(self.market_value.sum())


class StoredHoldings:
    """The sample books, opened once and read by date."""

    def __init__(self, root: Path | str = STORE_ROOT, directory: Path | None = None) -> None:
        self.directory = directory or Path(root) / PORTFOLIOS
        if not self.directory.is_dir():
            raise KeyError(f"no portfolios in {Path(root)}: build them first "
                           f"(scripts/build_sample_portfolios.py)")
        self.manifest = json.loads((self.directory / MANIFEST).read_text(encoding="utf-8"))
        self._holdings = _Dated(self.directory / f"{HOLDINGS}.parquet", "holdings_date", None)
        self.dates = tuple(sorted(self._holdings.index))
        self.portfolios = tuple(self.manifest.get("portfolios", ()))
        self._facts: dict[date, dict] | None = None

    def holdings(self, portfolio: str, day: date) -> Holdings:
        table = self._holdings.read(day)
        mask = np.array(table.column("portfolio").to_pylist()) == portfolio
        if not mask.any():
            raise KeyError(f"no {portfolio!r} holdings on {day} (books: "
                           f"{', '.join(self.portfolios)})")
        rows = table.filter(pa.array(mask))
        return Holdings(portfolio, day, tuple(rows.column("security").to_pylist()),
                        tuple(rows.column("security_key").to_pylist()),
                        tuple(rows.column("ticker").to_pylist()),
                        rows.column("quantity").to_numpy(), rows.column("price").to_numpy(),
                        rows.column("market_value").to_numpy(),
                        rows.column("weight").to_numpy())

    def facts(self, day: date) -> dict:
        if self._facts is None:
            table = pq.read_table(self.directory / f"{PORTFOLIO_DATES}.parquet")
            self._facts = {row["holdings_date"]: row for row in table.to_pylist()}
        return dict(self._facts[day])

    def all_facts(self) -> dict[date, dict]:
        self.facts(self.dates[0])
        return dict(self._facts or {})

    def close(self) -> None:
        self._holdings.close()


def list_models(root: Path | str = STORE_ROOT) -> list[str]:
    """The models in the store: each directory with a manifest, not a partial build."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and p.name not in (PARTIAL, PORTFOLIOS)
                  and (p / MANIFEST).exists())


def open_model(model_id: str, root: Path | str = STORE_ROOT) -> StoredModel:
    return StoredModel(root, model_id)
