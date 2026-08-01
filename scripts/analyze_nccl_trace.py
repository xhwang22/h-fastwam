#!/usr/bin/env python3
"""Summarize NCCL/compute overlap from a PyTorch Chrome trace on the CLI."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable

Interval = tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path, help="Path to step.json or step.json.gz.")
    return parser.parse_args()


def load_trace(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    events = payload.get("traceEvents", payload) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise ValueError("Trace must contain a list of traceEvents.")
    return [event for event in events if isinstance(event, dict)]


def event_interval(event: dict) -> Interval | None:
    if event.get("ph") != "X":
        return None
    try:
        start = float(event["ts"])
        duration = float(event["dur"])
    except (KeyError, TypeError, ValueError):
        return None
    if duration <= 0:
        return None
    return start, start + duration


def is_cuda_kernel(event: dict) -> bool:
    category = str(event.get("cat", "")).lower()
    args = event.get("args", {})
    arg_keys = {str(key).lower() for key in args} if isinstance(args, dict) else set()
    return "kernel" in category or {"device", "stream"}.issubset(arg_keys)


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    ordered = sorted(intervals)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = last_start, max(last_end, end)
        else:
            merged.append((start, end))
    return merged


def intersect_intervals(left: Iterable[Interval], right: Iterable[Interval]) -> list[Interval]:
    a = merge_intervals(left)
    b = merge_intervals(right)
    result: list[Interval] = []
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if start < end:
            result.append((start, end))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return result


def clip_intervals(intervals: Iterable[Interval], window: Interval) -> list[Interval]:
    return intersect_intervals(intervals, [window])


def duration(intervals: Iterable[Interval]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def ms(microseconds: float) -> float:
    return microseconds / 1000.0


def percentage(value: float, total: float) -> float:
    return 100.0 * value / total if total > 0 else 0.0


def main() -> None:
    args = parse_args()
    events = load_trace(args.trace)

    backward_candidates = []
    for event in events:
        if event.get("name") == "fastwam_backward":
            interval = event_interval(event)
            if interval is not None:
                backward_candidates.append(interval)
    if not backward_candidates:
        raise ValueError(
            "No fastwam_backward range found. Update to a version containing profiler timeline labels."
        )
    backward = max(backward_candidates, key=lambda item: item[1] - item[0])
    backward_duration = backward[1] - backward[0]

    cuda_events: list[tuple[str, Interval]] = []
    for event in events:
        interval = event_interval(event)
        if interval is None or not is_cuda_kernel(event):
            continue
        name = str(event.get("name", ""))
        cuda_events.append((name, interval))

    nccl = clip_intervals(
        (interval for name, interval in cuda_events if "nccl" in name.lower()),
        backward,
    )
    compute = clip_intervals(
        (interval for name, interval in cuda_events if "nccl" not in name.lower()),
        backward,
    )
    gemm_markers = ("gemm", "cutlass", "cublas", "tensorop", "ampere_", "hopper_")
    gemm = clip_intervals(
        (
            interval
            for name, interval in cuda_events
            if "nccl" not in name.lower()
            and any(marker in name.lower() for marker in gemm_markers)
        ),
        backward,
    )

    nccl_time = duration(nccl)
    compute_time = duration(compute)
    gemm_time = duration(gemm)
    nccl_compute_overlap = duration(intersect_intervals(nccl, compute))
    nccl_gemm_overlap = duration(intersect_intervals(nccl, gemm))
    exposed_nccl = max(nccl_time - nccl_compute_overlap, 0.0)

    tail_nccl = 0.0
    merged_compute = merge_intervals(compute)
    if merged_compute:
        last_compute_end = min(max(end for _, end in merged_compute), backward[1])
        if last_compute_end < backward[1]:
            tail_nccl = duration(
                intersect_intervals(nccl, [(last_compute_end, backward[1])])
            )

    print(f"trace={args.trace}")
    print(f"backward wall range:             {ms(backward_duration):10.3f} ms")
    print(
        f"NCCL union inside backward:      {ms(nccl_time):10.3f} ms "
        f"({percentage(nccl_time, backward_duration):6.2f}%)"
    )
    print(
        f"non-NCCL CUDA compute union:     {ms(compute_time):10.3f} ms "
        f"({percentage(compute_time, backward_duration):6.2f}%)"
    )
    print(
        f"GEMM/Tensor kernel union:        {ms(gemm_time):10.3f} ms "
        f"({percentage(gemm_time, backward_duration):6.2f}%)"
    )
    print(
        f"NCCL overlap with any compute:   {ms(nccl_compute_overlap):10.3f} ms "
        f"({percentage(nccl_compute_overlap, nccl_time):6.2f}% of NCCL)"
    )
    print(
        f"NCCL overlap with GEMM:          {ms(nccl_gemm_overlap):10.3f} ms "
        f"({percentage(nccl_gemm_overlap, nccl_time):6.2f}% of NCCL)"
    )
    print(
        f"NCCL without CUDA compute:       {ms(exposed_nccl):10.3f} ms "
        f"({percentage(exposed_nccl, backward_duration):6.2f}% of backward)"
    )
    print(
        f"NCCL after last compute kernel:  {ms(tail_nccl):10.3f} ms "
        f"({percentage(tail_nccl, backward_duration):6.2f}% of backward)"
    )

    if not nccl:
        print("result: no NCCL CUDA kernels were found inside fastwam_backward.")
    elif exposed_nccl > 0.25 * backward_duration:
        print("result: substantial NCCL time is not hidden by CUDA compute.")
    elif nccl_compute_overlap > 0.75 * nccl_time:
        print("result: most NCCL activity overlaps other CUDA compute.")
    else:
        print("result: NCCL is partially overlapped; inspect exposed and tail times.")


if __name__ == "__main__":
    main()
