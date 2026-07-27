#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import argparse
import os
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Node:
    name: str
    command: list[str]
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]
    needs: tuple[str, ...] = ()


def rel(path: str) -> Path:
    return ROOT / path


def artifact(output_dir: Path | None, path: str) -> Path:
    if output_dir is None:
        return rel(path)
    if not output_dir.is_absolute():
        output_dir = rel(str(output_dir))
    return output_dir / path


def local_path(path: Path | None, default: str) -> Path:
    resolved = path or Path(default)
    return resolved if resolved.is_absolute() else rel(str(resolved))


def regression_artifacts(
    output_dir: Path | None,
    regressor: str,
    arima_group_level: bool = False,
    arima_group_by: str = "cluster",
) -> tuple[Path, Path, Path]:
    stem = (
        (
            "grouped_arima_time_series_zone_level"
            if arima_group_by == "zone"
            else "grouped_arima_time_series_group_level"
        )
        if regressor == "arima" and arima_group_level
        else f"grouped_{regressor}_time_series"
    )
    return (
        artifact(output_dir, f"models/{stem}.joblib"),
        artifact(output_dir, f"{stem}_metrics.csv"),
        artifact(
            output_dir,
            (
                (
                    (
                        "predictions/arima_test_predictions_by_zone.parquet"
                        if arima_group_by == "zone"
                        else "predictions/arima_test_predictions_grouped.parquet"
                    )
                    if arima_group_level
                    else "predictions/arima_test_predictions_per_civic.parquet"
                )
                if regressor == "arima"
                else f"predictions/{regressor}_test_predictions_grouped.parquet"
            ),
        ),
    )


def arima_future_artifact(
    output_dir: Path | None,
    group_level: bool = False,
    group_by: str = "cluster",
) -> Path:
    return artifact(
        output_dir,
        (
            (
                "predictions/arima_future_predictions_by_zone.parquet"
                if group_by == "zone"
                else "predictions/arima_future_predictions_grouped.parquet"
            )
            if group_level
            else "predictions/arima_future_predictions_per_civic.parquet"
        ),
    )


def regression_command(
    python: str,
    regressor: str,
    gpr_max_train_rows: int,
    classified: Path,
    predictions: Path,
    model: Path,
    metrics: Path,
    seed: str,
    forecast_from_date: str | None = None,
    forecast_to_date: str | None = None,
    forecast_calendar: Path | None = None,
    future_predictions: Path | None = None,
    arima_group_level: bool = False,
    arima_group_by: str = "cluster",
) -> list[str]:
    if regressor == "arima":
        command = [
            python,
            "model-training/regression/train_grouped_arima_time_series.py",
            "--input",
            str(classified),
            "--split-ratios",
            "80,10,10",
            "--predictions-output",
            str(predictions),
            "--model-output",
            str(model),
            "--metrics-output",
            str(metrics),
        ]
        if arima_group_level:
            command.append("--group-level")
        if arima_group_by == "zone":
            command += ["--group-by", "zone"]
        if forecast_from_date:
            command += [
                "--forecast-from-date",
                forecast_from_date,
                "--forecast-to-date",
                forecast_to_date,
                "--future-predictions-output",
                str(future_predictions),
            ]
            if not arima_group_level:
                command += ["--forecast-calendar", str(forecast_calendar)]
        return command

    return [
        python,
        "model-training/regression/train_grouped_sgd_time_series_regressor.py",
        "--regressor",
        regressor,
        "--gpr-max-train-rows",
        str(gpr_max_train_rows),
        "--input",
        str(classified),
        "--split-ratios",
        "80,10,10",
        "--epochs",
        "3",
        "--batch-size",
        "4096",
        "--predictions-output",
        str(predictions),
        "--model-output",
        str(model),
        "--metrics-output",
        str(metrics),
        "--seed",
        seed,
        "--progress",
    ]


def normalize_training_range(from_date: str, to_date: str) -> tuple[str, str]:
    def normalize(value: str, end: bool) -> str:
        try:
            parsed_date = date.fromisoformat(value)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"Invalid ISO 8601 date: {value}") from exc
            if parsed.tzinfo is None:
                raise ValueError(f"Date must include a timezone: {value}")
            return value
        return f"{parsed_date.isoformat()}T{'23:59:59' if end else '00:00:00'}Z"

    start = normalize(from_date, False)
    end = normalize(to_date, True)
    if datetime.fromisoformat(start.replace("Z", "+00:00")) > datetime.fromisoformat(
        end.replace("Z", "+00:00")
    ):
        raise ValueError("--from-date must not be after --to-date")
    return start, end


def build_graph(
    python: str,
    seed: str,
    anno: str,
    output_dir: Path | None = None,
    annual_input: Path | None = None,
    calendar_input: Path | None = None,
    regressor: str = "sgd",
    gpr_max_train_rows: int = 500,
    forecast_from_date: str | None = None,
    forecast_to_date: str | None = None,
    forecast_calendar: Path | None = None,
    arima_group_level: bool = False,
    arima_group_by: str = "cluster",
) -> dict[str, Node]:
    annual_input = local_path(
        annual_input,
        "data_csv_parquet/andria_distribuzione_annuale_zone_inferite_e_zone_costruzione.csv",
    )
    calendar_input = local_path(
        calendar_input, "data_csv_parquet/calendario_raccolta.csv"
    )
    assignments = artifact(
        output_dir, "data_csv_parquet/kmeans_assignments_annuo_peso_coord.csv"
    )
    classified = artifact(
        output_dir, "data_csv_parquet/conferimenti_giornalieri_classificati.csv"
    )
    production = artifact(
        output_dir, "data_csv_parquet/produzione_giornaliera_classificata.csv"
    )
    classifier_model = artifact(
        output_dir, "models/kmeans_model_annuo_peso_coord.joblib"
    )
    regressor_model, metrics, predictions = regression_artifacts(
        output_dir, regressor, arima_group_level, arima_group_by
    )
    regression_node = f"grouped_{regressor}"
    future_predictions = (
        arima_future_artifact(output_dir, arima_group_level, arima_group_by)
        if regressor == "arima" and forecast_from_date
        else None
    )
    selected_forecast_calendar = (
        local_path(forecast_calendar, str(calendar_input))
        if future_predictions and not arima_group_level
        else None
    )
    regression_outputs = (regressor_model, metrics, predictions) + (
        (future_predictions,) if future_predictions else ()
    )
    graph = {
        "kmeans": Node(
            name="kmeans",
            command=[
                python,
                "model-training/classification/train_kmeans.py",
                "--input",
                str(annual_input),
                "--n-clusters",
                "6",
                "--output",
                str(assignments),
                "--output-format",
                "csv",
                "--model-output",
                str(classifier_model),
                "--coord-weight",
                "8",
                "--annual-kg-weight",
                "0.5",
                "--other-numeric-weight",
                "1",
                "--categorical-weight",
                "1",
                "--seed",
                seed,
            ],
            inputs=(annual_input,),
            outputs=(
                assignments,
                classifier_model,
            ),
        ),
        "daily_distribution": Node(
            name="daily_distribution",
            command=[
                python,
                "dataset/distribuzione_giornaliera.py",
                "--annuo",
                str(assignments),
                "--anno",
                anno,
                "--calendario",
                str(calendar_input),
                "--id-col",
                "id_civico",
                "--zona-col",
                "zona_raccolta",
                "--seed",
                seed,
                "--out",
                str(classified),
                "--out-produzione",
                str(production),
                "--missing-calendar-mode",
                "daily",
            ],
            inputs=(
                assignments,
                calendar_input,
            ),
            outputs=(classified, production),
            needs=("kmeans",),
        ),
        regression_node: Node(
            name=regression_node,
            command=regression_command(
                python,
                regressor,
                gpr_max_train_rows,
                classified,
                predictions,
                regressor_model,
                metrics,
                seed,
                forecast_from_date,
                forecast_to_date,
                selected_forecast_calendar,
                future_predictions,
                arima_group_level,
                arima_group_by,
            ),
            inputs=(classified,) + (
                (selected_forecast_calendar,) if selected_forecast_calendar else ()
            ),
            outputs=regression_outputs,
            needs=("daily_distribution",),
        ),
    }
    return graph


def build_complete_parquet_graph(
    python: str,
    model_seed: str,
    year: str,
    output_dir: Path,
    annual_input: Path,
    production_input: Path,
    collections_input: Path,
    calendar_input: Path,
    annual_seed: int = 42,
    scenario_seed: int = 42,
    spillover_days: int = 14,
    missing_calendar_mode: str = "error",
    generator_version: str = "unversioned-current",
    exclude_episodic_fractions: bool = False,
    hgb_loss: str = "poisson",
    include_arima: bool = False,
    validation_strategy: str = "fixed",
    validation_folds: int = 5,
) -> dict[str, Node]:
    output_dir = local_path(output_dir, str(output_dir))
    annual_input = local_path(annual_input, str(annual_input))
    production_input = local_path(production_input, str(production_input))
    collections_input = local_path(collections_input, str(collections_input))
    calendar_input = local_path(calendar_input, str(calendar_input))
    assignments = output_dir / "data" / "kmeans_assignments.parquet"
    classified_collections = (
        output_dir / "data" / "conferimenti_giornalieri_classificati.parquet"
    )
    classifier_model = output_dir / "models" / "kmeans_model.joblib"
    withdrawals = (
        output_dir
        / "data"
        / "withdrawals"
        / f"scenario_seed={scenario_seed}"
        / "part-000.parquet"
    )
    static_civics = output_dir / "data" / "static_civics.parquet"
    manifest = output_dir / "manifest.json"
    model = output_dir / "models" / "zone_fraction_benchmark.joblib"
    metrics = output_dir / "metrics" / "zone_fraction_benchmark_metrics.csv"
    predictions = (
        output_dir / "predictions" / "zone_fraction_test_predictions.parquet"
    )
    arima_model = output_dir / "models" / "zone_fraction_arima.joblib"
    arima_metrics = output_dir / "metrics" / "zone_fraction_arima_metrics.csv"
    arima_predictions = (
        output_dir
        / "predictions"
        / "zone_fraction_arima_test_predictions.parquet"
    )
    arima_civic_predictions = (
        output_dir
        / "predictions"
        / "zone_fraction_arima_test_predictions_per_civic.parquet"
    )
    arima_civic_metrics = (
        output_dir / "metrics" / "zone_fraction_arima_civic_metrics.csv"
    )
    arima_allocator_metrics = (
        output_dir / "metrics" / "zone_fraction_arima_allocator_benchmark.csv"
    )
    script = "model-training/zone_fraction_benchmark.py"
    graph = {
        "kmeans": Node(
            name="kmeans",
            command=[
                python,
                "model-training/classification/train_kmeans.py",
                "--input",
                str(annual_input),
                "--n-clusters",
                "6",
                "--output",
                str(assignments),
                "--output-format",
                "parquet",
                "--model-output",
                str(classifier_model),
                "--coord-weight",
                "8",
                "--annual-kg-weight",
                "0.5",
                "--other-numeric-weight",
                "1",
                "--categorical-weight",
                "1",
                "--seed",
                model_seed,
            ],
            inputs=(annual_input,),
            outputs=(assignments, classifier_model),
        ),
        "classify_collections": Node(
            name="classify_collections",
            command=[
                python,
                script,
                "classify",
                "--collections-input",
                str(collections_input),
                "--assignments-input",
                str(assignments),
                "--output",
                str(classified_collections),
            ],
            inputs=(collections_input, assignments),
            outputs=(classified_collections,),
            needs=("kmeans",),
        ),
        "prepare_zone_fraction": Node(
            name="prepare_zone_fraction",
            command=[
                python,
                script,
                "prepare",
                "--annual-input",
                str(annual_input),
                "--production-input",
                str(production_input),
                "--collections-input",
                str(classified_collections),
                "--calendar-input",
                str(calendar_input),
                "--output-dir",
                str(output_dir),
                "--year",
                year,
                "--annual-seed",
                str(annual_seed),
                "--scenario-seed",
                str(scenario_seed),
                "--spillover-days",
                str(spillover_days),
                "--missing-calendar-mode",
                missing_calendar_mode,
                "--generator-version",
                generator_version,
            ],
            inputs=(
                annual_input,
                production_input,
                classified_collections,
                calendar_input,
            ),
            outputs=(withdrawals, static_civics, manifest),
            needs=("classify_collections",),
        ),
        "train_zone_fraction": Node(
            name="train_zone_fraction",
            command=[
                python,
                script,
                "train",
                "--input",
                str(withdrawals),
                "--output-dir",
                str(output_dir),
                "--seed",
                model_seed,
                "--hgb-loss",
                hgb_loss,
                "--validation-strategy",
                validation_strategy,
                "--validation-folds",
                str(validation_folds),
                *(
                    ["--exclude-episodic-fractions"]
                    if exclude_episodic_fractions
                    else []
                ),
            ],
            inputs=(withdrawals, manifest),
            outputs=(model, metrics, predictions),
            needs=("prepare_zone_fraction",),
        ),
    }
    if include_arima:
        graph["train_zone_fraction_arima"] = Node(
            name="train_zone_fraction_arima",
            command=[
                python,
                script,
                "arima",
                "--input",
                str(withdrawals),
                "--civic-input",
                str(classified_collections),
                "--output-dir",
                str(output_dir),
                "--validation-strategy",
                validation_strategy,
                "--validation-folds",
                str(validation_folds),
                *(
                    ["--exclude-episodic-fractions"]
                    if exclude_episodic_fractions
                    else []
                ),
            ],
            inputs=(withdrawals, classified_collections, manifest),
            outputs=(
                arima_model,
                arima_metrics,
                arima_predictions,
                arima_civic_predictions,
                arima_civic_metrics,
                arima_allocator_metrics,
            ),
            needs=("prepare_zone_fraction",),
        )
    return graph


def build_quantumleap_graph(
    python: str,
    seed: str,
    temp_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Node]:
    daily = temp_dir / "quantumleap_daily.parquet"
    annual = temp_dir / "quantumleap_annual.parquet"
    assignments = temp_dir / "quantumleap_assignments.parquet"
    classified = temp_dir / "quantumleap_classified.parquet"
    output_dir = getattr(args, "output_dir", None)
    regressor = getattr(args, "regressor", "sgd")
    gpr_max_train_rows = getattr(args, "gpr_max_train_rows", 500)
    arima_group_level = getattr(args, "arima_group_level", False)
    arima_group_by = getattr(args, "arima_group_by", "cluster")
    classifier_model = artifact(
        output_dir, "models/kmeans_model_annuo_peso_coord.joblib"
    )
    regressor_model, metrics, predictions = regression_artifacts(
        output_dir, regressor, arima_group_level, arima_group_by
    )
    regression_node = f"grouped_{regressor}"
    forecast_from_date = getattr(args, "forecast_from_date", None)
    forecast_to_date = getattr(args, "forecast_to_date", None)
    forecast_calendar = getattr(args, "forecast_calendar", None)
    if forecast_calendar:
        forecast_calendar = Path(forecast_calendar)
        if not forecast_calendar.is_absolute():
            forecast_calendar = rel(str(forecast_calendar))
    future_predictions = (
        arima_future_artifact(output_dir, arima_group_level, arima_group_by)
        if regressor == "arima" and forecast_from_date
        else None
    )
    regression_outputs = (regressor_model, metrics, predictions) + (
        (future_predictions,) if future_predictions else ()
    )
    fetch_command = [
        python,
        "model-training/prepare_quantumleap.py",
        "fetch",
        "--output",
        str(daily),
        "--from-date",
        args.from_date,
        "--to-date",
        args.to_date,
        "--ql-url",
        args.ql_url,
        "--ql-entity-type",
        args.ql_entity_type,
        "--fiware-service",
        args.fiware_service,
        "--fiware-service-path",
        args.fiware_service_path,
        "--ql-timeout-seconds",
        str(args.ql_timeout_seconds),
    ]
    if args.ql_id_pattern:
        fetch_command += ["--ql-id-pattern", args.ql_id_pattern]

    graph = {
        "quantumleap_fetch": Node(
            "quantumleap_fetch", fetch_command, (), (daily,)
        ),
        "annual_aggregation": Node(
            "annual_aggregation",
            [
                python,
                "model-training/prepare_quantumleap.py",
                "aggregate",
                "--input",
                str(daily),
                "--output",
                str(annual),
            ],
            (daily,),
            (annual,),
            ("quantumleap_fetch",),
        ),
        "kmeans": Node(
            "kmeans",
            [
                python,
                "model-training/classification/train_kmeans.py",
                "--input",
                str(annual),
                "--n-clusters",
                "6",
                "--output",
                str(assignments),
                "--output-format",
                "parquet",
                "--model-output",
                str(classifier_model),
                "--coord-weight",
                "8",
                "--annual-kg-weight",
                "0.5",
                "--other-numeric-weight",
                "1",
                "--categorical-weight",
                "1",
                "--seed",
                seed,
            ],
            (annual,),
            (assignments, classifier_model),
            ("annual_aggregation",),
        ),
        "cluster_join": Node(
            "cluster_join",
            [
                python,
                "model-training/prepare_quantumleap.py",
                "join",
                "--daily",
                str(daily),
                "--assignments",
                str(assignments),
                "--output",
                str(classified),
            ],
            (daily, assignments),
            (classified,),
            ("kmeans",),
        ),
        regression_node: Node(
            regression_node,
            regression_command(
                python,
                regressor,
                gpr_max_train_rows,
                classified,
                predictions,
                regressor_model,
                metrics,
                seed,
                forecast_from_date,
                forecast_to_date,
                forecast_calendar,
                future_predictions,
                arima_group_level,
                arima_group_by,
            ),
            (classified,) + (
                (forecast_calendar,)
                if future_predictions and not arima_group_level
                else ()
            ),
            regression_outputs,
            ("cluster_join",),
        ),
    }
    if getattr(args, "publish_orion", False):
        calendar = Path(args.forecast_calendar)
        if not calendar.is_absolute():
            calendar = rel(args.forecast_calendar)
        stage = temp_dir / "digital_twins.jsonl"
        graph["publish_digital_twins"] = Node(
            "publish_digital_twins",
            [
                python,
                "model-training/publish_digital_twins.py",
                "--assignments",
                str(assignments),
                "--classifier-model",
                str(classifier_model),
                "--regressor-model",
                str(regressor_model),
                "--history",
                str(classified),
                "--calendar",
                str(calendar),
                "--forecast-from-date",
                args.forecast_from_date,
                "--forecast-to-date",
                args.forecast_to_date,
                "--training-to-date",
                args.to_date,
                "--stage-output",
                str(stage),
                "--orion-url",
                args.orion_url,
                "--orion-entity-type",
                args.orion_entity_type,
                "--orion-timeout-seconds",
                str(args.orion_timeout_seconds),
                "--orion-batch-size",
                str(args.orion_batch_size),
                "--fiware-service",
                args.fiware_service,
                "--fiware-service-path",
                args.fiware_service_path,
            ],
            (
                assignments,
                classifier_model,
                regressor_model,
                classified,
                calendar,
            ),
            (),
            (regression_node,),
        )
    return graph


def topo_sort(graph: dict[str, Node]) -> list[Node]:
    ordered = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise ValueError(f"Cycle in graph at node: {name}")
        if name not in graph:
            raise ValueError(f"Unknown dependency: {name}")

        visiting.add(name)
        for dep in graph[name].needs:
            visit(dep)
        visiting.remove(name)
        visited.add(name)
        ordered.append(graph[name])

    for name in graph:
        visit(name)

    return ordered


def is_current(node: Node) -> bool:
    if (
        not node.outputs
        or any(not path.exists() for path in node.inputs)
        or any(not output.exists() for output in node.outputs)
    ):
        return False

    newest_input = max((path.stat().st_mtime for path in node.inputs), default=0)
    oldest_output = min(output.stat().st_mtime for output in node.outputs)
    return oldest_output >= newest_input


def run_graph(graph: dict[str, Node], force: bool, dry_run: bool) -> None:
    graph_outputs = {output for node in graph.values() for output in node.outputs}

    for node in topo_sort(graph):
        missing_inputs = [
            str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
            for path in node.inputs
            if not path.exists() and (not dry_run or path not in graph_outputs)
        ]
        if missing_inputs:
            raise FileNotFoundError(
                f"{node.name} missing inputs: {', '.join(missing_inputs)}"
            )

        if not force and is_current(node):
            print(f"skip {node.name}: outputs current")
            continue

        print(f"run {node.name}: {' '.join(node.command)}")
        if not dry_run:
            subprocess.run(node.command, cwd=ROOT, check=True)


def print_mermaid(graph: dict[str, Node]) -> None:
    print("flowchart TD")
    for node in topo_sort(graph):
        print(f"    {node.name}[{node.name}]")
        for dep in node.needs:
            print(f"    {dep} --> {node.name}")


def self_check() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "source.txt"
        middle = tmp_path / "middle.txt"
        final = tmp_path / "final.txt"
        source.write_text("source", encoding="utf-8")

        graph = {
            "middle": Node(
                name="middle",
                command=[
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"Path({str(middle)!r}).write_text('middle')"
                    ),
                ],
                inputs=(source,),
                outputs=(middle,),
            ),
            "final": Node(
                name="final",
                command=[
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"Path({str(final)!r}).write_text('final')"
                    ),
                ],
                inputs=(middle,),
                outputs=(final,),
                needs=("middle",),
            ),
        }

        run_graph(graph, force=False, dry_run=False)
        assert middle.exists()
        assert final.exists()
        assert [node.name for node in topo_sort(graph)] == ["middle", "final"]
        assert is_current(graph["final"])

    print("self-check ok")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the README ML pipeline as a file DAG."
    )
    parser.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    parser.add_argument("--seed", default=os.environ.get("SEED", "42"))
    parser.add_argument("--anno", default=os.environ.get("ANNO", "2024"))
    parser.add_argument(
        "--regressor",
        choices=["sgd", "gpr", "hist_gradient_boosting", "arima"],
        default="sgd",
    )
    parser.add_argument("--gpr-max-train-rows", type=int, default=500)
    parser.add_argument("--arima-group-level", action="store_true")
    parser.add_argument(
        "--arima-group-by", choices=["cluster", "zone"], default="cluster"
    )
    parser.add_argument(
        "--source",
        choices=["file", "quantumleap", "complete-parquet"],
        default="file",
    )
    parser.add_argument(
        "--annual-input",
        type=Path,
        default=Path(
            "data_csv_parquet/andria_distribuzione_annuale_completa.parquet"
        ),
    )
    parser.add_argument(
        "--production-input",
        type=Path,
        default=Path("data_csv_parquet/produzione_giornaliera_completa.parquet"),
    )
    parser.add_argument(
        "--collections-input",
        type=Path,
        default=Path("data_csv_parquet/conferimenti_giornalieri_completi.parquet"),
    )
    parser.add_argument("--annual-seed", type=int, default=42)
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--spillover-days", type=int, default=14)
    parser.add_argument(
        "--missing-calendar-mode", choices=["error", "daily"], default="error"
    )
    parser.add_argument("--generator-version", default="unversioned-current")
    parser.add_argument("--exclude-episodic-fractions", action="store_true")
    parser.add_argument(
        "--hgb-loss",
        choices=["poisson", "squared_error"],
        default="poisson",
    )
    parser.add_argument(
        "--validation-strategy",
        choices=["fixed", "k-fold"],
        default="fixed",
    )
    parser.add_argument("--validation-folds", type=int, default=5)
    parser.add_argument("--include-arima", action="store_true")
    parser.add_argument(
        "--calendar-input",
        type=Path,
        default=Path("data_csv_parquet/calendario_raccolta.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Write generated artifacts under this directory instead of the repository."
        ),
    )
    parser.add_argument("--from-date")
    parser.add_argument("--to-date")
    parser.add_argument("--ql-id-pattern")
    parser.add_argument(
        "--ql-url", default=os.environ.get("QL_URL", "http://localhost:8668")
    )
    parser.add_argument(
        "--ql-entity-type",
        default=os.environ.get("ENTITY_TYPE", "RifiutiConferimento"),
    )
    parser.add_argument(
        "--fiware-service", default=os.environ.get("FIWARE_SERVICE", "openiot")
    )
    parser.add_argument(
        "--fiware-service-path",
        default=os.environ.get("FIWARE_SERVICE_PATH", "/"),
    )
    parser.add_argument(
        "--ql-timeout-seconds",
        type=float,
        default=float(os.environ.get("QL_TIMEOUT_SECONDS", "30")),
    )
    parser.add_argument("--publish-orion", action="store_true")
    parser.add_argument("--forecast-from-date")
    parser.add_argument("--forecast-to-date")
    parser.add_argument(
        "--forecast-calendar",
        default=os.environ.get(
            "FORECAST_CALENDAR", "data_csv_parquet/calendario_raccolta.csv"
        ),
    )
    parser.add_argument(
        "--orion-url", default=os.environ.get("ORION_URL", "http://localhost:1026")
    )
    parser.add_argument(
        "--orion-entity-type",
        default=os.environ.get("ORION_TWIN_ENTITY_TYPE", "RifiutiDigitalTwin"),
    )
    parser.add_argument(
        "--orion-timeout-seconds",
        type=float,
        default=float(os.environ.get("ORION_TIMEOUT_SECONDS", "30")),
    )
    parser.add_argument(
        "--orion-batch-size",
        type=int,
        default=int(os.environ.get("ORION_BATCH_SIZE", "20")),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-graph", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return

    if args.gpr_max_train_rows <= 0:
        parser.error("--gpr-max-train-rows must be greater than zero")
    if args.spillover_days < 0:
        parser.error("--spillover-days must be non-negative")
    if args.validation_strategy == "k-fold" and args.validation_folds < 2:
        parser.error("--validation-folds must be at least 2")

    if bool(args.forecast_from_date) != bool(args.forecast_to_date):
        parser.error(
            "--forecast-from-date and --forecast-to-date must be supplied together"
        )
    if args.regressor == "arima" and args.publish_orion:
        parser.error("ARIMA supports batch files, not --publish-orion")
    if args.arima_group_level and args.regressor != "arima":
        parser.error("--arima-group-level requires --regressor arima")
    if args.arima_group_by == "zone" and not (
        args.regressor == "arima" and args.arima_group_level
    ):
        parser.error(
            "--arima-group-by zone requires --regressor arima "
            "and --arima-group-level"
        )
    if (
        args.forecast_from_date
        and args.regressor != "arima"
        and not args.publish_orion
    ):
        parser.error("forecast dates without --publish-orion require --regressor arima")

    if args.source == "quantumleap":
        if not args.from_date or not args.to_date:
            parser.error(
                "--from-date and --to-date are required for --source quantumleap"
            )
        try:
            args.from_date, args.to_date = normalize_training_range(
                args.from_date, args.to_date
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.ql_timeout_seconds <= 0:
            parser.error("--ql-timeout-seconds must be greater than zero")

    if args.publish_orion:
        if args.source != "quantumleap":
            parser.error("--publish-orion requires --source quantumleap")
        if not args.forecast_from_date or not args.forecast_to_date:
            parser.error(
                "--forecast-from-date and --forecast-to-date are required "
                "with --publish-orion"
            )
        try:
            forecast_start = date.fromisoformat(args.forecast_from_date)
            forecast_end = date.fromisoformat(args.forecast_to_date)
            training_end = datetime.fromisoformat(
                args.to_date.replace("Z", "+00:00")
            ).date()
        except (AttributeError, ValueError) as exc:
            parser.error(f"Invalid publication date: {exc}")
        if forecast_start > forecast_end:
            parser.error("--forecast-from-date must not be after --forecast-to-date")
        if forecast_start <= training_end:
            parser.error("forecast dates must be after --to-date")
        if args.orion_timeout_seconds <= 0 or args.orion_batch_size <= 0:
            parser.error("Orion timeout and batch size must be greater than zero")

    if args.source == "complete-parquet":
        if args.output_dir is None:
            parser.error("--output-dir is required for --source complete-parquet")
        graph = build_complete_parquet_graph(
            args.python,
            args.seed,
            args.anno,
            args.output_dir,
            args.annual_input,
            args.production_input,
            args.collections_input,
            args.calendar_input,
            args.annual_seed,
            args.scenario_seed,
            args.spillover_days,
            args.missing_calendar_mode,
            args.generator_version,
            args.exclude_episodic_fractions,
            args.hgb_loss,
            args.include_arima,
            args.validation_strategy,
            args.validation_folds,
        )
        if args.print_graph:
            print_mermaid(graph)
            return
        run_graph(graph, force=args.force, dry_run=args.dry_run)
        return

    if args.source == "file":
        graph = build_graph(
            args.python,
            args.seed,
            args.anno,
            args.output_dir,
            calendar_input=args.calendar_input,
            regressor=args.regressor,
            gpr_max_train_rows=args.gpr_max_train_rows,
            forecast_from_date=args.forecast_from_date,
            forecast_to_date=args.forecast_to_date,
            forecast_calendar=Path(args.forecast_calendar),
            arima_group_level=args.arima_group_level,
            arima_group_by=args.arima_group_by,
        )
        if args.print_graph:
            print_mermaid(graph)
            return
        run_graph(graph, force=args.force, dry_run=args.dry_run)
        return

    with tempfile.TemporaryDirectory(prefix="quantumleap-training-") as tmp:
        graph = build_quantumleap_graph(args.python, args.seed, Path(tmp), args)
        if args.print_graph:
            print_mermaid(graph)
            return
        run_graph(graph, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
