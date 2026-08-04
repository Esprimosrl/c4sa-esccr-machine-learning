#!/usr/bin/env python3

from __future__ import annotations

from itertools import islice
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
import argparse
import math
import os
import resource
import sys
import tempfile
import time
import traceback

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepare_quantumleap import DAILY_ATTRS, fetch_to_parquet, write_parquet
from quantumleap import PAGE_SIZE, QuantumLeapClient


def client_for(config: dict[str, object]) -> QuantumLeapClient:
    return QuantumLeapClient(
        str(config["ql_url"]),
        str(config["entity_type"]),
        str(config["fiware_service"]),
        str(config["fiware_service_path"]),
        float(config["timeout_seconds"]),
    )


def page_benchmark(
    config: dict[str, object], concurrency: int, requested_pages: int
) -> dict[str, float | int]:
    pages = client_for(config).fetch_pages(
        list(DAILY_ATTRS),
        str(config["from_date"]),
        str(config["to_date"]),
        id_pattern=config["id_pattern"],
        concurrency=concurrency,
    )
    page_count = 0
    row_count = 0
    start = time.perf_counter()
    try:
        for page in islice(pages, requested_pages):
            page_count += 1
            row_count += len(page)
    finally:
        pages.close()
    elapsed = time.perf_counter() - start
    return {
        "workers": concurrency,
        "pages": page_count,
        "rows": row_count,
        "elapsed_seconds": elapsed,
        "pages_per_second": page_count / elapsed,
        "rows_per_second": row_count / elapsed,
    }


def end_to_end_worker(
    mode: str,
    config: dict[str, object],
    output: str,
    result_queue,
) -> None:
    try:
        client = client_for(config)
        output_path = Path(output)
        start = time.perf_counter()
        if mode == "legacy":
            rows = client.fetch(
                list(DAILY_ATTRS),
                str(config["from_date"]),
                str(config["to_date"]),
                id_pattern=config["id_pattern"],
            )
            if not rows:
                raise ValueError("No QuantumLeap records matched the benchmark range.")
            frame = pd.DataFrame(rows)
            frame["data"] = frame["timestamp"]
            write_parquet(frame, output_path)
        else:
            fetch_to_parquet(
                client,
                list(DAILY_ATTRS),
                str(config["from_date"]),
                str(config["to_date"]),
                output_path,
                config["id_pattern"],
            )
        elapsed = time.perf_counter() - start
        row_count = duckdb.connect().execute(
            "SELECT count(*) FROM read_parquet(?)", [str(output_path)]
        ).fetchone()[0]
        result_queue.put(
            {
                "mode": mode,
                "rows": row_count,
                "elapsed_seconds": elapsed,
                "rows_per_second": row_count / elapsed,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024,
                "output_bytes": os.path.getsize(output_path),
            }
        )
    except BaseException:
        result_queue.put({"mode": mode, "error": traceback.format_exc()})
        raise


def run_end_to_end(
    mode: str,
    config: dict[str, object],
    output: Path,
) -> dict[str, object]:
    context = get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=end_to_end_worker,
        args=(mode, config, str(output), result_queue),
    )
    process.start()
    process.join()
    try:
        result = result_queue.get(timeout=1)
    except Empty as exc:
        raise RuntimeError(f"{mode} benchmark exited without a result.") from exc
    finally:
        result_queue.close()
    if process.exitcode != 0 or "error" in result:
        raise RuntimeError(result.get("error", f"{mode} benchmark failed."))
    return result


def outputs_match(left: Path, right: Path) -> bool:
    con = duckdb.connect()
    try:
        left_only = con.execute(
            """
            SELECT count(*) FROM (
                SELECT * FROM read_parquet(?)
                EXCEPT ALL
                SELECT * FROM read_parquet(?)
            )
            """,
            [str(left), str(right)],
        ).fetchone()[0]
        right_only = con.execute(
            """
            SELECT count(*) FROM (
                SELECT * FROM read_parquet(?)
                EXCEPT ALL
                SELECT * FROM read_parquet(?)
            )
            """,
            [str(right), str(left)],
        ).fetchone()[0]
        return left_only == 0 and right_only == 0
    finally:
        con.close()


def print_page_result(result: dict[str, float | int]) -> None:
    print(
        f"workers={result['workers']} pages={result['pages']} "
        f"rows={result['rows']} elapsed={result['elapsed_seconds']:.3f}s "
        f"pages/s={result['pages_per_second']:.3f} "
        f"rows/s={result['rows_per_second']:.1f}"
    )


def print_full_result(result: dict[str, object]) -> None:
    print(
        f"{result['mode']}: rows={result['rows']} "
        f"elapsed={result['elapsed_seconds']:.2f}s "
        f"rows/s={result['rows_per_second']:.1f} "
        f"peak_rss={result['peak_rss_mib']:.1f} MiB "
        f"output={result['output_bytes'] / 1_000_000:.2f} MB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare sequential and two-worker streamed QuantumLeap acquisition."
        )
    )
    parser.add_argument("--ql-url", default="http://localhost:8668")
    parser.add_argument("--entity-type", default="RifiutiConferimento")
    parser.add_argument("--fiware-service", default="openiot")
    parser.add_argument("--fiware-service-path", default="/")
    parser.add_argument("--timeout-seconds", type=float, default=30)
    parser.add_argument("--from-date", default="2024-01-01T00:00:00Z")
    parser.add_argument("--to-date", default="2024-01-02T23:59:59Z")
    parser.add_argument("--id-pattern")
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--project-rows", type=int, default=32_386_883)
    parser.add_argument("--page-payload-mb", type=float, default=12.8)
    args = parser.parse_args()
    if args.pages <= 0 or args.project_rows <= 0:
        parser.error("--pages and --project-rows must be greater than zero")

    config = {
        "ql_url": args.ql_url,
        "entity_type": args.entity_type,
        "fiware_service": args.fiware_service,
        "fiware_service_path": args.fiware_service_path,
        "timeout_seconds": args.timeout_seconds,
        "from_date": args.from_date,
        "to_date": args.to_date,
        "id_pattern": args.id_pattern,
    }

    print("Page throughput")
    sequential = page_benchmark(config, 1, args.pages)
    concurrent = page_benchmark(config, 2, args.pages)
    print_page_result(sequential)
    print_page_result(concurrent)
    print(
        f"page speedup: "
        f"{concurrent['rows_per_second'] / sequential['rows_per_second']:.2f}x"
    )

    print("\nEnd-to-end acquisition")
    with tempfile.TemporaryDirectory(prefix="quantumleap-benchmark-") as temp_dir:
        temp = Path(temp_dir)
        legacy_path = temp / "legacy.parquet"
        streamed_path = temp / "streamed.parquet"
        legacy = run_end_to_end("legacy", config, legacy_path)
        streamed = run_end_to_end("streamed", config, streamed_path)
        print_full_result(legacy)
        print_full_result(streamed)
        print(
            f"end-to-end speedup: "
            f"{legacy['elapsed_seconds'] / streamed['elapsed_seconds']:.2f}x"
        )
        identical = outputs_match(legacy_path, streamed_path)
        print(f"outputs identical: {'yes' if identical else 'no'}")

    projected_pages = math.ceil(args.project_rows / PAGE_SIZE)
    projected_hours = (
        projected_pages / float(concurrent["pages_per_second"]) / 3600
    )
    projected_gb = projected_pages * args.page_payload_mb / 1000
    print(
        f"\nProjection for {args.project_rows:,} rows: "
        f"{projected_pages:,} pages, {projected_hours:.2f} hours, "
        f"{projected_gb:.1f} GB JSON"
    )


if __name__ == "__main__":
    main()
