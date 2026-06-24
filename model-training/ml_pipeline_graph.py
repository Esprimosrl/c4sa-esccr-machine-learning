#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
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


def build_graph(python: str, seed: str, anno: str) -> dict[str, Node]:
    return {
        "kmeans": Node(
            name="kmeans",
            command=[
                python,
                "model-training/classification/train_kmeans.py",
                "--input",
                "data_csv_parquet/andria_distribuzione_annuale_zone_inferite_e_zone_costruzione.csv",
                "--n-clusters",
                "6",
                "--output",
                "data_csv_parquet/kmeans_assignments_annuo_peso_coord.csv",
                "--output-format",
                "csv",
                "--model-output",
                "models/kmeans_model_annuo_peso_coord.joblib",
                "--coord-weight",
                "8",
                "--annual-kg-weight",
                "0.5",
                "--other-numeric-weight",
                "1",
                "--categorical-weight",
                "1",
            ],
            inputs=(
                rel(
                    "data_csv_parquet/andria_distribuzione_annuale_zone_inferite_e_zone_costruzione.csv"
                ),
            ),
            outputs=(
                rel("data_csv_parquet/kmeans_assignments_annuo_peso_coord.csv"),
                rel("models/kmeans_model_annuo_peso_coord.joblib"),
            ),
        ),
        "daily_distribution": Node(
            name="daily_distribution",
            command=[
                python,
                "dataset/distribuzione_giornaliera.py",
                "--annuo",
                "data_csv_parquet/kmeans_assignments_annuo_peso_coord.csv",
                "--anno",
                anno,
                "--calendario",
                "data_csv_parquet/calendario_raccolta.csv",
                "--id-col",
                "id_civico",
                "--zona-col",
                "zona_raccolta",
                "--seed",
                seed,
                "--out",
                "data_csv_parquet/conferimenti_giornalieri_classificati.csv",
                "--out-produzione",
                "data_csv_parquet/produzione_giornaliera_classificata.csv",
                "--missing-calendar-mode",
                "daily",
            ],
            inputs=(
                rel("data_csv_parquet/kmeans_assignments_annuo_peso_coord.csv"),
                rel("data_csv_parquet/calendario_raccolta.csv"),
            ),
            outputs=(
                rel("data_csv_parquet/conferimenti_giornalieri_classificati.csv"),
                rel("data_csv_parquet/produzione_giornaliera_classificata.csv"),
            ),
            needs=("kmeans",),
        ),
        "grouped_sgd": Node(
            name="grouped_sgd",
            command=[
                python,
                "model-training/regression/train_grouped_sgd_time_series_regressor.py",
                "--input",
                "data_csv_parquet/conferimenti_giornalieri_classificati.csv",
                "--split-ratios",
                "80,10,10",
                "--epochs",
                "3",
                "--batch-size",
                "4096",
                "--predictions-output",
                "predictions/sgd_test_predictions_grouped.parquet",
                "--progress",
            ],
            inputs=(rel("data_csv_parquet/conferimenti_giornalieri_classificati.csv"),),
            outputs=(
                rel("models/grouped_sgd_time_series.joblib"),
                rel("grouped_sgd_time_series_metrics.csv"),
                rel("predictions/sgd_test_predictions_grouped.parquet"),
            ),
            needs=("daily_distribution",),
        ),
    }


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
    if not node.outputs or any(not output.exists() for output in node.outputs):
        return False

    newest_input = max((path.stat().st_mtime for path in node.inputs), default=0)
    oldest_output = min(output.stat().st_mtime for output in node.outputs)
    return oldest_output >= newest_input


def run_graph(graph: dict[str, Node], force: bool, dry_run: bool) -> None:
    graph_outputs = {output for node in graph.values() for output in node.outputs}

    for node in topo_sort(graph):
        missing_inputs = [
            str(path.relative_to(ROOT))
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-graph", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return

    graph = build_graph(args.python, args.seed, args.anno)
    if args.print_graph:
        print_mermaid(graph)
        return

    run_graph(graph, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
