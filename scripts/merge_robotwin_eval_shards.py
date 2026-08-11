#!/usr/bin/env python3
"""Merge task-disjoint RoboTwin manager shard summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def merge(run_output_dir: Path, shard_count: int) -> None:
    per_task: dict[str, dict] = {}
    failed_lines = []
    for shard_index in range(shard_count):
        summary_path = (
            run_output_dir / f"summary.shard{shard_index}-of-{shard_count}.json"
        )
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing shard summary: {summary_path}")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for record in payload["per_task"]:
            task_name = str(record["task_name"])
            if task_name in per_task:
                raise ValueError(f"Duplicate task across shards: {task_name}")
            per_task[task_name] = record
        failed_path = (
            run_output_dir / f"failed_tasks.shard{shard_index}-of-{shard_count}.txt"
        )
        if failed_path.is_file():
            failed_lines.extend(failed_path.read_text(encoding="utf-8").splitlines())

    records = [per_task[name] for name in sorted(per_task)]
    clean_values = [
        float(record["clean_success_rate"])
        for record in records
        if record["clean_success_rate"] is not None
    ]
    random_values = [
        float(record["random_success_rate"])
        for record in records
        if record["random_success_rate"] is not None
    ]
    overall = {
        "clean_mean_success_rate": _mean(clean_values),
        "random_mean_success_rate": _mean(random_values),
    }

    summary_json = run_output_dir / "summary.json"
    summary_json.write_text(
        json.dumps({"per_task": records, "overall": overall}, indent=2),
        encoding="utf-8",
    )
    with (run_output_dir / "summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_name", "clean_success_rate", "random_success_rate"])
        for record in records:
            writer.writerow(
                [
                    record["task_name"],
                    record["clean_success_rate"],
                    record["random_success_rate"],
                ]
            )
        writer.writerow(
            [
                "__overall__",
                overall["clean_mean_success_rate"],
                overall["random_mean_success_rate"],
            ]
        )
    (run_output_dir / "failed_tasks.txt").write_text(
        "".join(f"{line}\n" for line in failed_lines if line),
        encoding="utf-8",
    )
    print(f"Merged {shard_count} shards and {len(records)} tasks: {run_output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-output-dir", required=True)
    parser.add_argument("--shard-count", required=True, type=int)
    args = parser.parse_args()
    merge(Path(args.run_output_dir).expanduser().resolve(), args.shard_count)


if __name__ == "__main__":
    main()
