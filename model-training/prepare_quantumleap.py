#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantumleap import QuantumLeapClient


FRACTIONS = (
    "altro_rd",
    "carta_cartone",
    "frazione_organica",
    "ingombranti_misti",
    "legno",
    "metallo",
    "plastica",
    "pulizia_stradale_recupero",
    "raee",
    "rifiuti_cd",
    "selettiva",
    "tessili",
    "vetro",
)
DAILY_ATTRS = (
    "id_civico",
    "frazione",
    "anno_produzione",
    "coord_x",
    "coord_y",
    "densita_locale",
    "zona_raccolta",
    "zona_costruzione",
    "giorni_accumulati",
    "conferimento_t",
)
STATIC_COLUMNS = (
    "coord_x",
    "coord_y",
    "densita_locale",
    "zona_raccolta",
    "zona_costruzione",
)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.register("frame", frame)
    con.execute("COPY frame TO ? (FORMAT PARQUET)", [str(path)])
    con.close()


def annual_rows(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"entity_id", "timestamp", *DAILY_ATTRS}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"Missing daily columns: {', '.join(sorted(missing))}.")
    daily = daily.copy()
    for column in ("id_civico", "frazione", "entity_id"):
        daily[column] = daily[column].astype("string")
    for column in (
        "anno_produzione",
        "coord_x",
        "coord_y",
        "densita_locale",
        "zona_raccolta",
        "giorni_accumulati",
        "conferimento_t",
    ):
        try:
            daily[column] = pd.to_numeric(daily[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric values in {column}.") from exc
    unknown = set(daily["frazione"].unique()) - set(FRACTIONS)
    if unknown:
        raise ValueError(f"Unknown fractions: {', '.join(sorted(map(str, unknown)))}.")

    keys = ["id_civico", "anno_produzione"]
    for column in STATIC_COLUMNS:
        conflicts = daily.groupby(keys, dropna=False)[column].nunique(dropna=False) > 1
        if conflicts.any():
            raise ValueError(f"Conflicting {column} within civic/year.")
    entity_conflicts = (
        daily.groupby(keys + ["frazione"], dropna=False)["entity_id"]
        .nunique(dropna=False)
        .gt(1)
    )
    if entity_conflicts.any():
        raise ValueError("Conflicting entity ID within civic/year/fraction.")
    fraction_conflicts = (
        daily.groupby(keys + ["entity_id"], dropna=False)["frazione"]
        .nunique(dropna=False)
        .gt(1)
    )
    if fraction_conflicts.any():
        raise ValueError("Conflicting fraction within civic/year/entity ID.")

    totals = daily.pivot_table(
        index=keys,
        columns="frazione",
        values="conferimento_t",
        aggfunc="sum",
        fill_value=0,
    ).reindex(columns=FRACTIONS, fill_value=0)
    totals.columns = [f"{column}_t" for column in totals.columns]
    static = daily.groupby(keys, dropna=False)[list(STATIC_COLUMNS)].first()
    result = static.join(totals).reset_index()
    fraction_columns = [f"{name}_t" for name in FRACTIONS]
    result["totale_t"] = result[fraction_columns].sum(axis=1)
    return result


def join_clusters(daily: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    year_col = (
        "anno_from_col" if "anno_from_col" in assignments else "anno_produzione"
    )
    needed = {"id_civico", year_col, "cluster_label"}
    if not needed <= set(assignments.columns):
        raise ValueError(
            "Assignments missing columns: "
            f"{', '.join(sorted(needed - set(assignments.columns)))}."
        )
    labels = assignments[["id_civico", year_col, "cluster_label"]].rename(
        columns={year_col: "anno_produzione"}
    ).copy()
    classified = daily.drop(columns=["cluster_label"], errors="ignore").copy()
    for frame in (classified, labels):
        frame["id_civico"] = frame["id_civico"].astype("string")
        frame["anno_produzione"] = pd.to_numeric(
            frame["anno_produzione"], errors="raise"
        ).astype("Int64")
    if labels.duplicated(["id_civico", "anno_produzione"]).any():
        raise ValueError("K-means returned duplicate civic/year assignments.")
    classified = classified.merge(
        labels,
        on=["id_civico", "anno_produzione"],
        how="left",
        validate="many_to_one",
    )
    if classified["cluster_label"].isna().any() or len(classified) != len(daily):
        raise ValueError("Every daily row must have exactly one fresh cluster assignment.")
    history_groups = classified.groupby(
        ["id_civico", "anno_produzione", "frazione"], dropna=False
    )["conferimento_t"]
    classified["historical_annual_t"] = history_groups.transform("sum")
    classified["historical_collection_mean_t"] = history_groups.transform("mean")
    return classified


def read_parquet(path: Path) -> pd.DataFrame:
    return duckdb.connect().execute("SELECT * FROM read_parquet(?)", [str(path)]).df()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--from-date", required=True)
    fetch.add_argument("--to-date", required=True)
    fetch.add_argument("--ql-url", required=True)
    fetch.add_argument("--ql-entity-type", required=True)
    fetch.add_argument("--fiware-service", required=True)
    fetch.add_argument("--fiware-service-path", required=True)
    fetch.add_argument("--ql-timeout-seconds", type=float, required=True)
    fetch.add_argument("--ql-id-pattern")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--input", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    join = subparsers.add_parser("join")
    join.add_argument("--daily", type=Path, required=True)
    join.add_argument("--assignments", type=Path, required=True)
    join.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "fetch":
        client = QuantumLeapClient(
            args.ql_url,
            args.ql_entity_type,
            args.fiware_service,
            args.fiware_service_path,
            args.ql_timeout_seconds,
        )
        rows = client.fetch(
            list(DAILY_ATTRS), args.from_date, args.to_date, id_pattern=args.ql_id_pattern
        )
        if not rows:
            raise ValueError("No QuantumLeap records matched the training range.")
        frame = pd.DataFrame(rows)
        frame["data"] = frame["timestamp"]
        write_parquet(frame, args.output)
    elif args.command == "aggregate":
        write_parquet(annual_rows(read_parquet(args.input)), args.output)
    else:
        write_parquet(
            join_clusters(read_parquet(args.daily), read_parquet(args.assignments)),
            args.output,
        )


if __name__ == "__main__":
    main()
