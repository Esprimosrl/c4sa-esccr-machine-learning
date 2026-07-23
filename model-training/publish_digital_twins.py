#!/usr/bin/env python3

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from urllib.request import urlopen
import argparse
import hashlib
import json
import math
import sys

import duckdb
import joblib
import numpy as np
import pandas as pd

from lag_features import LAG_FEATURES, LagState

ROOT = Path(__file__).resolve().parents[1]
REGRESSION_DIR = ROOT / "model-training" / "regression"
for path in (ROOT, REGRESSION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset.distribuzione_giornaliera import (
    FRACTIONS,
    build_collection_map,
    read_calendar_events,
)
from train_grouped_sgd_time_series_regressor import (
    clip_raw_predictions,
    make_features,
    prediction_log_to_raw,
)
from orion import publish_entities


CIVIC_BATCH_SIZE = 1_000


def parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid date {value!r}; expected YYYY-MM-DD.") from exc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_zone(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def split_years(start: date, end: date) -> list[tuple[int, date, date]]:
    return [
        (
            year,
            max(start, date(year, 1, 1)),
            min(end, date(year, 12, 31)),
        )
        for year in range(start.year, end.year + 1)
    ]


def latest_assignments(assignments: pd.DataFrame, forecast_year: int) -> pd.DataFrame:
    year_col = (
        "anno_from_col" if "anno_from_col" in assignments else "anno_produzione"
    )
    required = {
        "id_civico",
        year_col,
        "coord_x",
        "coord_y",
        "densita_locale",
        "zona_raccolta",
        "cluster_label",
        "cluster_distance",
    }
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(
            f"Assignments missing columns: {', '.join(sorted(missing))}."
        )

    history_columns = {
        column for column in assignments.columns if column.endswith("_t")
    }
    eligible = assignments[sorted(required | history_columns)].copy()
    eligible[year_col] = pd.to_numeric(eligible[year_col], errors="raise")
    eligible = eligible[eligible[year_col] <= forecast_year]
    if eligible.empty:
        raise ValueError(
            f"No cluster assignments are available at or before {forecast_year}."
        )
    eligible["id_civico"] = eligible["id_civico"].astype("string")
    eligible["zona_raccolta"] = eligible["zona_raccolta"].map(normalize_zone)
    eligible["cluster_label"] = pd.to_numeric(
        eligible["cluster_label"], errors="raise"
    ).astype(int)
    eligible = (
        eligible.sort_values(["id_civico", year_col])
        .drop_duplicates("id_civico", keep="last")
        .rename(columns={year_col: "classification_year"})
        .reset_index(drop=True)
    )
    if eligible[list(required - {year_col})].isna().any().any():
        raise ValueError("Cluster assignments contain null required values.")
    return eligible


def collection_features(
    calendar_path: Path,
    zones: list[str],
    year: int,
    start: date,
    end: date,
) -> pd.DataFrame:
    events = read_calendar_events(
        path=str(calendar_path),
        separator=",",
        zones=zones,
        year=year,
        spillover_days=14,
    )
    rows = []
    for fraction in FRACTIONS:
        collection_map = build_collection_map(
            calendar_events=events,
            zones=zones,
            fraction=fraction,
            year=year,
            spillover_days=14,
            missing_calendar_mode="daily",
            force_year_end=True,
        )
        counts: dict[tuple[str, date], int] = {}
        for item in collection_map.select(
            ["zona_raccolta", "conferimento_data"]
        ).to_dicts():
            event_date = item["conferimento_data"]
            if isinstance(event_date, str):
                event_date = date.fromisoformat(event_date)
            key = (normalize_zone(item["zona_raccolta"]), event_date)
            counts[key] = counts.get(key, 0) + 1
        collection_counts: dict[str, int] = {}
        for zone, _ in counts:
            collection_counts[zone] = collection_counts.get(zone, 0) + 1
        for (zone, event_date), accumulated in counts.items():
            if start <= event_date <= end and event_date.year == year:
                rows.append(
                    {
                        "zona_raccolta": zone,
                        "frazione": fraction.label,
                        "data": event_date,
                        "giorni_accumulati": accumulated,
                        "annual_collection_count": collection_counts[zone],
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "zona_raccolta",
            "frazione",
            "data",
            "giorni_accumulati",
            "annual_collection_count",
        ],
    ).sort_values(["data", "zona_raccolta", "frazione"])


def add_date_features(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    result = frame.copy()
    dates = pd.to_datetime(result["data"])
    day_of_year = dates.dt.dayofyear
    result["anno_ref"] = year
    result["month"] = dates.dt.month
    result["day_of_month"] = dates.dt.day
    result["day_of_year"] = day_of_year
    result["day_of_week"] = (dates.dt.dayofweek + 1) % 7
    result["week_of_year"] = dates.map(lambda value: int(value.strftime("%W")))
    result["quarter"] = dates.dt.quarter
    result["season_sin"] = np.sin(2 * math.pi * day_of_year / 366.0)
    result["season_cos"] = np.cos(2 * math.pi * day_of_year / 366.0)
    return result


def load_lag_state(history_path: Path, before: date) -> LagState:
    reader = "read_parquet(?)" if history_path.suffix != ".csv" else "read_csv_auto(?)"
    con = duckdb.connect()
    cursor = con.execute(f"""
        WITH typed AS (
            SELECT
                CAST(id_civico AS VARCHAR) AS id_civico,
                CAST(frazione AS VARCHAR) AS frazione,
                CAST(data AS DATE) AS data_dt,
                TRY_CAST(conferimento_t AS DOUBLE) AS target_value
            FROM {reader}
        ),
        past AS (
            SELECT *
            FROM typed
            WHERE data_dt < CAST(? AS DATE)
              AND target_value IS NOT NULL
        ),
        selected AS (
            SELECT *
            FROM past
            WHERE data_dt >= CAST(? AS DATE) - INTERVAL 90 DAY
            UNION
            SELECT *
            FROM past
            QUALIFY row_number() OVER (
                PARTITION BY id_civico, frazione ORDER BY data_dt DESC
            ) = 1
            UNION
            SELECT *
            FROM past
            WHERE target_value > 0
            QUALIFY row_number() OVER (
                PARTITION BY id_civico, frazione ORDER BY data_dt DESC
            ) = 1
        )
        SELECT id_civico, frazione, data_dt, target_value
        FROM selected
        ORDER BY id_civico, frazione, data_dt;
    """, [str(history_path), before, before])
    state = LagState()
    while rows := cursor.fetchmany(100_000):
        for civic, fraction, event_date, value in rows:
            state.add(civic, fraction, event_date, value)
    con.close()
    return state


def typed_attribute(value: object, ngsi_type: str) -> dict[str, object]:
    return {"type": ngsi_type, "value": value}


def build_entity(
    assignment: pd.Series,
    forecast_year: int,
    forecast_start: date,
    forecast_end: date,
    values: dict[date, dict[str, float]],
    entity_type: str,
    classifier_version: str,
    regressor_version: str,
    observed_at: str,
) -> dict[str, object]:
    civic = str(assignment["id_civico"])
    future_values = [
        {
            "date": event_date.isoformat(),
            "values_t": {
                fraction.label: float(fractions[fraction.label])
                for fraction in FRACTIONS
                if fraction.label in fractions
            },
        }
        for event_date, fractions in sorted(values.items())
    ]
    return {
        "id": f"urn:ngsi-v2:{entity_type}:{civic}:{forecast_year}",
        "type": entity_type,
        "id_civico": typed_attribute(civic, "Text"),
        "forecast_year": typed_attribute(forecast_year, "Integer"),
        "classification_year": typed_attribute(
            int(assignment["classification_year"]), "Integer"
        ),
        "cluster_label": typed_attribute(
            int(assignment["cluster_label"]), "Integer"
        ),
        "cluster_distance": typed_attribute(
            float(assignment["cluster_distance"]), "Number"
        ),
        "classifier_model_version": typed_attribute(classifier_version, "Text"),
        "regressor_model_version": typed_attribute(regressor_version, "Text"),
        "forecast_from": typed_attribute(
            f"{forecast_start.isoformat()}T00:00:00Z", "DateTime"
        ),
        "forecast_to": typed_attribute(
            f"{forecast_end.isoformat()}T23:59:59Z", "DateTime"
        ),
        "dateObserved": typed_attribute(observed_at, "DateTime"),
        "future_values": typed_attribute(future_values, "StructuredValue"),
    }


def stage_twins(
    assignments_path: Path,
    classifier_path: Path,
    regressor_path: Path,
    calendar_path: Path,
    forecast_start: date,
    forecast_end: date,
    entity_type: str,
    output_path: Path,
    history_path: Path | None = None,
) -> int:
    if not entity_type.strip():
        raise ValueError("Orion entity type must not be empty.")
    assignments = duckdb.connect().execute(
        "SELECT * FROM read_parquet(?)", [str(assignments_path)]
    ).df()
    bundle = joblib.load(regressor_path)
    if bundle.get("model_name") not in {
        "grouped_sgd",
        "grouped_gpr",
        "grouped_hist_gradient_boosting",
    }:
        raise ValueError("Unsupported grouped regressor bundle.")
    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]
    uses_lags = any(name in numeric_cols for name in LAG_FEATURES)
    if uses_lags and history_path is None:
        raise ValueError("Regressor lag features require a history dataset.")
    lag_state = load_lag_state(history_path, forecast_start) if uses_lags else None
    known_fractions = {item.label for item in FRACTIONS}
    bundle_fractions = {
        key.split("|", 1)[0].removeprefix("frazione=")
        for key in bundle["models"]
    }
    unknown_fractions = bundle_fractions - known_fractions
    if unknown_fractions:
        raise ValueError(
            "Regressor contains unknown fractions: "
            + ", ".join(sorted(unknown_fractions))
        )
    model_fractions = [
        item.label for item in FRACTIONS if item.label in bundle_fractions
    ]

    classifier_version = file_sha256(classifier_path)
    regressor_version = file_sha256(regressor_path)
    observed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entity_count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for year, year_start, year_end in split_years(
            forecast_start, forecast_end
        ):
            selected = latest_assignments(assignments, year)
            events = collection_features(
                calendar_path,
                sorted(selected["zona_raccolta"].unique()),
                year,
                year_start,
                year_end,
            )
            for offset in range(0, len(selected), CIVIC_BATCH_SIZE):
                civics = selected.iloc[offset : offset + CIVIC_BATCH_SIZE].copy()
                values: dict[str, dict[date, dict[str, float]]] = {
                    str(civic): {} for civic in civics["id_civico"]
                }
                for fraction in model_fractions:
                    fraction_events = events[events["frazione"] == fraction]
                    if fraction_events.empty:
                        continue
                    for cluster_label, group in civics.groupby("cluster_label"):
                        model_key = (
                            f"frazione={fraction}|cluster_label={int(cluster_label)}"
                        )
                        if model_key not in bundle["models"]:
                            raise ValueError(f"No regressor model for {model_key}.")
                        merged = group.merge(
                            fraction_events,
                            on="zona_raccolta",
                            how="inner",
                            validate="many_to_many",
                        )
                        if merged.empty:
                            continue
                        annual_col = f"{fraction}_t"
                        if "historical_annual_t" in numeric_cols:
                            if annual_col not in merged:
                                raise ValueError(
                                    f"Assignments missing historical quantity: {annual_col}."
                                )
                            merged["historical_annual_t"] = pd.to_numeric(
                                merged[annual_col], errors="raise"
                            )
                        if "historical_collection_mean_t" in numeric_cols:
                            merged["historical_collection_mean_t"] = (
                                pd.to_numeric(merged[annual_col], errors="raise")
                                / merged["annual_collection_count"]
                            )
                        merged = add_date_features(merged, year)
                        info = bundle["models"][model_key]
                        prediction_upper_bound = info.get(
                            "prediction_upper_bound_t"
                        )
                        for event_date, positions in merged.groupby(
                            "data", sort=True
                        ).indices.items():
                            batch = merged.iloc[positions].copy()
                            if lag_state is not None:
                                lag_rows = [
                                    lag_state.features(civic, fraction, event_date)
                                    for civic in batch["id_civico"]
                                ]
                                for name in LAG_FEATURES:
                                    batch[name] = [row[name] for row in lag_rows]
                            features = make_features(
                                batch,
                                numeric_cols,
                                categorical_cols,
                                info["numeric_stats"],
                            )
                            predictions = info["model"].predict(features)
                            if bundle.get("target_transform", "log1p") == "log1p":
                                predictions = prediction_log_to_raw(
                                    predictions, prediction_upper_bound
                                )
                            else:
                                predictions = clip_raw_predictions(
                                    predictions, prediction_upper_bound
                                )
                            for civic, prediction in zip(
                                batch["id_civico"], predictions, strict=True
                            ):
                                if lag_state is not None:
                                    lag_state.add(
                                        civic, fraction, event_date, float(prediction)
                                    )
                                values[str(civic)].setdefault(event_date, {})[
                                    fraction
                                ] = float(prediction)

                for _, assignment in civics.iterrows():
                    entity = build_entity(
                        assignment,
                        year,
                        year_start,
                        year_end,
                        values[str(assignment["id_civico"])],
                        entity_type,
                        classifier_version,
                        regressor_version,
                        observed_at,
                    )
                    output.write(json.dumps(entity, separators=(",", ":")) + "\n")
                    entity_count += 1
    return entity_count


def publish_twins(
    stage_path: Path,
    orion_url: str,
    fiware_service: str,
    fiware_service_path: str,
    timeout_seconds: float,
    batch_size: int,
) -> int:
    with stage_path.open(encoding="utf-8") as source:
        entities = [json.loads(line) for line in source]
    return publish_entities(
        entities,
        orion_url,
        fiware_service,
        fiware_service_path,
        timeout_seconds,
        batch_size,
        opener=urlopen,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate inferred civic/year twins and publish them to Orion."
    )
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--classifier-model", type=Path, required=True)
    parser.add_argument("--regressor-model", type=Path, required=True)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--forecast-from-date", required=True)
    parser.add_argument("--forecast-to-date", required=True)
    parser.add_argument("--training-to-date", required=True)
    parser.add_argument("--stage-output", type=Path, required=True)
    parser.add_argument("--orion-url", required=True)
    parser.add_argument("--orion-entity-type", required=True)
    parser.add_argument("--orion-timeout-seconds", type=float, required=True)
    parser.add_argument("--orion-batch-size", type=int, required=True)
    parser.add_argument("--fiware-service", required=True)
    parser.add_argument("--fiware-service-path", required=True)
    args = parser.parse_args()

    forecast_start = parse_day(args.forecast_from_date)
    forecast_end = parse_day(args.forecast_to_date)
    training_end = datetime.fromisoformat(
        args.training_to_date.replace("Z", "+00:00")
    ).date()
    if forecast_start > forecast_end:
        raise ValueError("forecast-from-date must not be after forecast-to-date.")
    if forecast_start <= training_end:
        raise ValueError("Forecast dates must be after the training range.")
    if args.orion_timeout_seconds <= 0 or args.orion_batch_size <= 0:
        raise ValueError("Orion timeout and batch size must be greater than zero.")

    staged = stage_twins(
        args.assignments,
        args.classifier_model,
        args.regressor_model,
        args.calendar,
        forecast_start,
        forecast_end,
        args.orion_entity_type,
        args.stage_output,
        args.history,
    )
    published = publish_twins(
        args.stage_output,
        args.orion_url,
        args.fiware_service,
        args.fiware_service_path,
        args.orion_timeout_seconds,
        args.orion_batch_size,
    )
    if published != staged:
        raise RuntimeError(f"Published {published} of {staged} staged twins.")
    print(f"published {published} {args.orion_entity_type} entities")


if __name__ == "__main__":
    main()
