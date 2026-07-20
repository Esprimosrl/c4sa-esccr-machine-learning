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
    regressor_model = artifact(
        output_dir, "models/grouped_sgd_time_series.joblib"
    )
    metrics = artifact(output_dir, "grouped_sgd_time_series_metrics.csv")
    predictions = artifact(
        output_dir, "predictions/sgd_test_predictions_grouped.parquet"
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
        "grouped_sgd": Node(
            name="grouped_sgd",
            command=[
                python,
                "model-training/regression/train_grouped_sgd_time_series_regressor.py",
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
                str(regressor_model),
                "--metrics-output",
                str(metrics),
                "--seed",
                seed,
                "--progress",
            ],
            inputs=(classified,),
            outputs=(regressor_model, metrics, predictions),
            needs=("daily_distribution",),
        ),
    }
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
    classifier_model = artifact(
        output_dir, "models/kmeans_model_annuo_peso_coord.joblib"
    )
    regressor_model = artifact(
        output_dir, "models/grouped_sgd_time_series.joblib"
    )
    metrics = artifact(output_dir, "grouped_sgd_time_series_metrics.csv")
    predictions = artifact(
        output_dir, "predictions/sgd_test_predictions_grouped.parquet"
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
        "grouped_sgd": Node(
            "grouped_sgd",
            [
                python,
                "model-training/regression/train_grouped_sgd_time_series_regressor.py",
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
                str(regressor_model),
                "--metrics-output",
                str(metrics),
                "--seed",
                seed,
                "--progress",
            ],
            (classified,),
            (
                regressor_model,
                metrics,
                predictions,
            ),
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
                calendar,
            ),
            (),
            ("grouped_sgd",),
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
    parser.add_argument("--source", choices=["file", "quantumleap"], default="file")
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

    if args.source == "file":
        graph = build_graph(args.python, args.seed, args.anno, args.output_dir)
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
