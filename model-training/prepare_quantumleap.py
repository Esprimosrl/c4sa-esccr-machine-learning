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
COMPLETE_DAILY_ATTRS = DAILY_ATTRS + (
    "produzione_teorica_t",
    "conferimento_data",
    "raccolta_prevista",
    "prg_error",
    "zona_raccolta_inferita",
    "zona_costruzione_riassegnata",
    "distanza_inferenza_zona_raccolta_gradi",
    "distanza_riassegnazione_prg",
)
STATIC_COLUMNS = (
    "coord_x",
    "coord_y",
    "densita_locale",
    "zona_raccolta",
    "zona_costruzione",
)
COMPLETE_ANNUAL_STATIC_COLUMNS = STATIC_COLUMNS + (
    "prg_error",
    "zona_raccolta_inferita",
    "zona_costruzione_riassegnata",
    "distanza_inferenza_zona_raccolta_gradi",
    "distanza_riassegnazione_prg",
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


def complete_snapshots(
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"entity_id", "timestamp", *COMPLETE_DAILY_ATTRS}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(
            "QuantumLeap complete history is missing columns: "
            + ", ".join(sorted(missing))
            + "."
        )

    frame = daily.copy()
    frame["data"] = pd.to_datetime(frame["timestamp"], errors="raise").dt.date
    frame["id_civico"] = frame["id_civico"].astype("string")
    frame["frazione"] = frame["frazione"].astype("string")
    numeric = (
        "anno_produzione",
        "coord_x",
        "coord_y",
        "densita_locale",
        "zona_raccolta",
        "giorni_accumulati",
        "conferimento_t",
        "produzione_teorica_t",
        "raccolta_prevista",
        "zona_raccolta_inferita",
        "zona_costruzione_riassegnata",
        "distanza_inferenza_zona_raccolta_gradi",
        "distanza_riassegnazione_prg",
    )
    for column in numeric:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric values in {column}.") from exc
    frame["conferimento_data"] = pd.to_datetime(
        frame["conferimento_data"], errors="raise"
    ).dt.date
    unknown = set(frame["frazione"].dropna().unique()) - set(FRACTIONS)
    if unknown:
        raise ValueError(f"Unknown fractions: {', '.join(sorted(map(str, unknown)))}.")
    if frame[list(numeric)].isna().any().any():
        raise ValueError("QuantumLeap complete history contains null numeric values.")
    if (frame[["conferimento_t", "produzione_teorica_t"]] < 0).any().any():
        raise ValueError("QuantumLeap complete history contains negative quantities.")
    if not frame["raccolta_prevista"].isin([0, 1]).all():
        raise ValueError("raccolta_prevista must contain only 0 or 1.")

    keys = ["id_civico", "anno_produzione"]
    for column in COMPLETE_ANNUAL_STATIC_COLUMNS:
        conflicts = frame.groupby(keys, dropna=False)[column].nunique(dropna=False) > 1
        if conflicts.any():
            raise ValueError(f"Conflicting {column} within civic/year.")

    totals = frame.pivot_table(
        index=keys,
        columns="frazione",
        values="conferimento_t",
        aggfunc="sum",
        fill_value=0,
    ).reindex(columns=FRACTIONS, fill_value=0)
    totals.columns = [f"{column}_t" for column in totals.columns]
    static = frame.groupby(keys, dropna=False)[
        list(COMPLETE_ANNUAL_STATIC_COLUMNS)
    ].first()
    annual = static.join(totals).reset_index().rename(
        columns={"anno_produzione": "anno"}
    )
    fraction_columns = [f"{name}_t" for name in FRACTIONS]
    annual["totale_t"] = annual[fraction_columns].sum(axis=1)

    production = frame[
        [
            "anno_produzione",
            "data",
            "id_civico",
            "zona_raccolta",
            "frazione",
            "produzione_teorica_t",
            "conferimento_data",
            "raccolta_prevista",
        ]
    ].copy()
    collections = frame.loc[
        frame["conferimento_t"] > 0,
        [
            "anno_produzione",
            "data",
            "id_civico",
            "coord_x",
            "coord_y",
            "densita_locale",
            "zona_raccolta",
            "zona_costruzione",
            "frazione",
            "conferimento_t",
            "giorni_accumulati",
        ],
    ].copy()
    return annual, production, collections


def collection_snapshots(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual = annual_rows(daily).rename(columns={"anno_produzione": "anno"})
    collections = daily.loc[
        daily["conferimento_t"] > 0,
        [
            "anno_produzione",
            "data",
            "id_civico",
            "coord_x",
            "coord_y",
            "densita_locale",
            "zona_raccolta",
            "zona_costruzione",
            "frazione",
            "conferimento_t",
            "giorni_accumulati",
        ],
    ].copy()
    return annual, collections


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
    fetch.add_argument(
        "--profile", choices=["legacy", "complete"], default="legacy"
    )
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--input", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    join = subparsers.add_parser("join")
    join.add_argument("--daily", type=Path, required=True)
    join.add_argument("--assignments", type=Path, required=True)
    join.add_argument("--output", type=Path, required=True)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--input", type=Path, required=True)
    snapshot.add_argument("--annual-output", type=Path, required=True)
    snapshot.add_argument("--production-output", type=Path, required=True)
    snapshot.add_argument("--collections-output", type=Path, required=True)
    collections_snapshot = subparsers.add_parser("snapshot-collections")
    collections_snapshot.add_argument("--input", type=Path, required=True)
    collections_snapshot.add_argument("--annual-output", type=Path, required=True)
    collections_snapshot.add_argument(
        "--collections-output", type=Path, required=True
    )
    args = parser.parse_args()

    if args.command == "fetch":
        client = QuantumLeapClient(
            args.ql_url,
            args.ql_entity_type,
            args.fiware_service,
            args.fiware_service_path,
            args.ql_timeout_seconds,
        )
        attrs = COMPLETE_DAILY_ATTRS if args.profile == "complete" else DAILY_ATTRS
        rows = client.fetch(
            list(attrs), args.from_date, args.to_date, id_pattern=args.ql_id_pattern
        )
        if not rows:
            raise ValueError("No QuantumLeap records matched the training range.")
        frame = pd.DataFrame(rows)
        frame["data"] = frame["timestamp"]
        write_parquet(frame, args.output)
    elif args.command == "aggregate":
        write_parquet(annual_rows(read_parquet(args.input)), args.output)
    elif args.command == "join":
        write_parquet(
            join_clusters(read_parquet(args.daily), read_parquet(args.assignments)),
            args.output,
        )
    elif args.command == "snapshot":
        annual, production, collections = complete_snapshots(
            read_parquet(args.input)
        )
        write_parquet(annual, args.annual_output)
        write_parquet(production, args.production_output)
        write_parquet(collections, args.collections_output)
    else:
        annual, collections = collection_snapshots(read_parquet(args.input))
        write_parquet(annual, args.annual_output)
        write_parquet(collections, args.collections_output)


if __name__ == "__main__":
    main()
