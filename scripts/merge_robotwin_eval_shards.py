#!/usr/bin/env python3
"""Merge task-disjoint RoboTwin manager shard summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def merge(
    run_output_dir: Path,
    shard_count: int,
    *,
    allow_partial: bool = False,
) -> None:
    per_task: dict[str, dict] = {}
    failed_lines = []
    shards_found = []
    shards_missing = []
    for shard_index in range(shard_count):
        summary_path = (
            run_output_dir / f"summary.shard{shard_index}-of-{shard_count}.json"
        )
        if not summary_path.is_file():
            if allow_partial:
                shards_missing.append(shard_index)
                continue
            raise FileNotFoundError(f"Missing shard summary: {summary_path}")
        shards_found.append(shard_index)
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

    if not per_task:
        raise ValueError(
            f"No shard summaries found under {run_output_dir} "
            f"for shard_count={shard_count}."
        )
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
    progress = {
        "shard_count": shard_count,
        "shards_found": shards_found,
        "shards_missing": shards_missing,
        "tasks_seen": len(records),
        "clean_phases_completed": len(clean_values),
        "random_phases_completed": len(random_values),
        "tasks_fully_completed": sum(
            record["clean_success_rate"] is not None
            and record["random_success_rate"] is not None
            for record in records
        ),
        "tasks_with_any_result": sum(
            record["clean_success_rate"] is not None
            or record["random_success_rate"] is not None
            for record in records
        ),
    }

    output_suffix = ".partial" if allow_partial else ""
    summary_json = run_output_dir / f"summary{output_suffix}.json"
    summary_json.write_text(
        json.dumps(
            {
                "partial": allow_partial,
                "progress": progress,
                "per_task": records,
                "overall": overall,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (run_output_dir / f"summary{output_suffix}.csv").open(
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
    (run_output_dir / f"failed_tasks{output_suffix}.txt").write_text(
        "".join(f"{line}\n" for line in failed_lines if line),
        encoding="utf-8",
    )
    print(
        f"Merged {len(shards_found)}/{shard_count} shards and "
        f"{len(records)} task records: {run_output_dir}"
    )
    print(
        "Completed phases: "
        f"clean={len(clean_values)} random={len(random_values)} "
        f"full_tasks={progress['tasks_fully_completed']}"
    )
    if allow_partial:
        print(
            "Partial means include only completed phases and may be biased "
            "toward faster/easier tasks."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-output-dir", required=True)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    merge(
        Path(args.run_output_dir).expanduser().resolve(),
        args.shard_count,
        allow_partial=args.allow_partial,
    )


if __name__ == "__main__":
    main()
