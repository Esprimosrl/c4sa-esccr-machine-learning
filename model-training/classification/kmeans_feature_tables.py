#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import duckdb

REGRESSION_DIR = Path(__file__).resolve().parents[1] / "regression"
if str(REGRESSION_DIR) not in sys.path:
    sys.path.insert(0, str(REGRESSION_DIR))

from modular_hist_gradient_boosting_regressor import (
    categorical_expr,
    create_standard_table,
    numeric_expr,
    sql_ident,
    sql_lit,
    table_columns,
)


IGNORED_ANNUAL_T_COLUMNS = {"totale_t"}
KMEANS_EXTRA_CATEGORICAL_COLS = ("zona_costruzione",)


@dataclass(frozen=True)
class FeatureTableInfo:
    mode: str
    annual_date_mm_dd: str | None = None
    annual_fraction_columns: tuple[str, ...] = ()
    ignored_t_columns: tuple[str, ...] = ()


def resolve_column(
    cols: set[str],
    preferred: str,
    fallbacks: tuple[str, ...],
) -> str | None:
    if preferred in cols:
        return preferred

    for fallback in fallbacks:
        if fallback in cols:
            return fallback

    return None


def annual_fraction_columns(cols: set[str]) -> list[tuple[str, str]]:
    t_cols = sorted(
        col
        for col in cols
        if col.endswith("_t") and col not in IGNORED_ANNUAL_T_COLUMNS
    )
    return [(col.removesuffix("_t"), col) for col in t_cols]


def parse_annual_date_mm_dd(value: str) -> tuple[int, int]:
    parts = value.split("-")
    if len(parts) != 2:
        raise ValueError("--annual-date-mm-dd must use MM-DD format.")

    month = int(parts[0])
    day = int(parts[1])

    if month < 1 or month > 12:
        raise ValueError("--annual-date-mm-dd month must be between 1 and 12.")

    if day < 1 or day > 31:
        raise ValueError("--annual-date-mm-dd day must be between 1 and 31.")

    return month, day


def create_annual_wide_feature_table(
    con: duckdb.DuckDBPyConnection,
    raw_table: str,
    out_table: str,
    id_col: str,
    year_col: str,
    annual_date_mm_dd: str,
) -> FeatureTableInfo:
    cols = table_columns(con, raw_table)
    resolved_year_col = resolve_column(cols, year_col, ("anno", "anno_produzione"))
    resolved_zone_col = resolve_column(cols, "zona_raccolta", ("zona",))
    fraction_columns = annual_fraction_columns(cols)

    if id_col not in cols:
        raise ValueError(f"Missing id column: {id_col}")

    if resolved_year_col is None:
        raise ValueError(
            "Missing annual year column. Expected anno or anno_produzione."
        )

    if resolved_zone_col is None:
        raise ValueError("Missing zone column. Expected zona_raccolta or zona.")

    if not fraction_columns:
        raise ValueError("No annual *_t fraction columns were found.")

    month, day = parse_annual_date_mm_dd(annual_date_mm_dd)
    fraction_numeric_exprs = ",\n                ".join(
        numeric_expr(t_col, cols) for _, t_col in fraction_columns
    )

    if "totale_t" in cols:
        target_expr = """
            TRY_CAST(
                REPLACE(CAST("totale_t" AS VARCHAR), ',', '.')
                AS DOUBLE
            )
        """
    else:
        target_expr = "NULL::DOUBLE"

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {out_table} AS
        WITH typed AS (
            SELECT
                __row_id,
                CAST({sql_ident(id_col)} AS VARCHAR) AS id_civico,
                TRY_CAST({sql_ident(resolved_year_col)} AS INTEGER) AS anno_from_col,

                {target_expr} AS target_value,

                {numeric_expr("coord_x", cols)},
                {numeric_expr("coord_y", cols)},
                {numeric_expr("densita_locale", cols)},
                {numeric_expr("giorni_accumulati", cols)},
                {fraction_numeric_exprs},

                CAST({sql_ident(resolved_zone_col)} AS VARCHAR) AS zona_raccolta,
                {categorical_expr("zona_costruzione", cols)}
            FROM {raw_table}
        ),

        enriched AS (
            SELECT
                __row_id,
                id_civico,
                MAKE_DATE(anno_from_col, {month}, {day}) AS data_dt,
                anno_from_col,
                target_value,
                coord_x,
                coord_y,
                densita_locale,
                giorni_accumulati,
                {", ".join(sql_ident(t_col) for _, t_col in fraction_columns)},
                zona_raccolta,
                zona_costruzione,
                anno_from_col AS anno_ref,
                {month}::INTEGER AS month,
                {day}::INTEGER AS day_of_month,
                CAST(strftime(MAKE_DATE(anno_from_col, {month}, {day}), '%j') AS INTEGER)
                    AS day_of_year,
                CAST(strftime(MAKE_DATE(anno_from_col, {month}, {day}), '%w') AS INTEGER)
                    AS day_of_week,
                CAST(strftime(MAKE_DATE(anno_from_col, {month}, {day}), '%W') AS INTEGER)
                    AS week_of_year,
                EXTRACT(QUARTER FROM MAKE_DATE(anno_from_col, {month}, {day}))::INTEGER
                    AS quarter,
                sin(
                    2 * pi()
                    * CAST(strftime(MAKE_DATE(anno_from_col, {month}, {day}), '%j') AS DOUBLE)
                    / 366.0
                ) AS season_sin,
                cos(
                    2 * pi()
                    * CAST(strftime(MAKE_DATE(anno_from_col, {month}, {day}), '%j') AS DOUBLE)
                    / 366.0
                ) AS season_cos
            FROM typed
        )

        SELECT
            __row_id,
            id_civico,
            data_dt,
            anno_from_col,
            target_value,
            coord_x,
            coord_y,
            densita_locale,
            giorni_accumulati,
            {", ".join(sql_ident(t_col) for _, t_col in fraction_columns)},
            zona_raccolta,
            zona_costruzione,
            anno_ref,
            month,
            day_of_month,
            day_of_year,
            day_of_week,
            week_of_year,
            quarter,
            season_sin,
            season_cos
        FROM enriched;
    """)

    ignored = sorted(col for col in cols if col in IGNORED_ANNUAL_T_COLUMNS)

    return FeatureTableInfo(
        mode="annual_wide",
        annual_date_mm_dd=annual_date_mm_dd,
        annual_fraction_columns=tuple(col for _, col in fraction_columns),
        ignored_t_columns=tuple(ignored),
    )


def add_extra_kmeans_categorical_columns(
    con: duckdb.DuckDBPyConnection,
    raw_table: str,
    out_table: str,
) -> None:
    cols = table_columns(con, raw_table)
    selects = ",\n            ".join(
        (
            f"CAST(r.{sql_ident(col)} AS VARCHAR) AS {sql_ident(col)}"
            if col in cols
            else f"NULL::VARCHAR AS {sql_ident(col)}"
        )
        for col in KMEANS_EXTRA_CATEGORICAL_COLS
    )

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {out_table}_with_extra_categories AS
        SELECT
            f.*,
            {selects}
        FROM {out_table} f
        LEFT JOIN {raw_table} r
        ON f.__row_id = r.__row_id;
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {out_table} AS
        SELECT * FROM {out_table}_with_extra_categories;
    """)


def create_kmeans_feature_table(
    con: duckdb.DuckDBPyConnection,
    raw_table: str,
    out_table: str,
    id_col: str,
    date_col: str,
    year_col: str,
    target_col: str,
    date_format: str,
    annual_date_mm_dd: str,
) -> FeatureTableInfo:
    cols = table_columns(con, raw_table)

    if date_col in cols:
        create_standard_table(
            con=con,
            raw_table=raw_table,
            out_table=out_table,
            id_col=id_col,
            date_col=date_col,
            year_col=year_col,
            target_col=target_col,
            date_format=date_format,
            require_target=False,
        )
        add_extra_kmeans_categorical_columns(con, raw_table, out_table)
        return FeatureTableInfo(mode="date_based")

    return create_annual_wide_feature_table(
        con=con,
        raw_table=raw_table,
        out_table=out_table,
        id_col=id_col,
        year_col=year_col,
        annual_date_mm_dd=annual_date_mm_dd,
    )
