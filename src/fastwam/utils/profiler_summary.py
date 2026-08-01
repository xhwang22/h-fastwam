"""Utilities for summarizing NCCL overlap from an in-memory torch profile."""

from __future__ import annotations


def _merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _intersect_intervals(left, right):
    left = _merge_intervals(left)
    right = _merge_intervals(right)
    result = []
    left_idx = right_idx = 0
    while left_idx < len(left) and right_idx < len(right):
        start = max(left[left_idx][0], right[right_idx][0])
        end = min(left[left_idx][1], right[right_idx][1])
        if start < end:
            result.append((start, end))
        if left[left_idx][1] <= right[right_idx][1]:
            left_idx += 1
        else:
            right_idx += 1
    return result


def _duration(intervals):
    return sum(end - start for start, end in _merge_intervals(intervals))


def summarize_nccl_profile(torch_profile) -> str:
    backward_ranges = []
    cuda_events = []
    for event in torch_profile.events():
        time_range = getattr(event, "time_range", None)
        if time_range is None:
            continue
        start = float(time_range.start)
        end = float(time_range.end)
        if end <= start:
            continue
        name = str(getattr(event, "name", ""))
        device_type = str(getattr(event, "device_type", "")).lower()
        if name == "fastwam_backward":
            backward_ranges.append((start, end))
        if "cuda" in device_type:
            cuda_events.append((name, (start, end)))

    if not backward_ranges:
        return "No fastwam_backward range was found."
    backward = max(backward_ranges, key=lambda item: item[1] - item[0])
    backward_duration = backward[1] - backward[0]

    def clip(intervals):
        return _intersect_intervals(intervals, [backward])

    nccl = clip(interval for name, interval in cuda_events if "nccl" in name.lower())
    compute = clip(interval for name, interval in cuda_events if "nccl" not in name.lower())
    gemm_markers = ("gemm", "cutlass", "cublas", "tensorop", "ampere_", "hopper_")
    gemm = clip(
        interval
        for name, interval in cuda_events
        if "nccl" not in name.lower()
        and any(marker in name.lower() for marker in gemm_markers)
    )

    nccl_time = _duration(nccl)
    compute_time = _duration(compute)
    gemm_time = _duration(gemm)
    nccl_compute_overlap = _duration(_intersect_intervals(nccl, compute))
    nccl_gemm_overlap = _duration(_intersect_intervals(nccl, gemm))
    exposed_nccl = max(nccl_time - nccl_compute_overlap, 0.0)

    tail_nccl = 0.0
    merged_compute = _merge_intervals(compute)
    if merged_compute:
        last_compute_end = min(max(end for _, end in merged_compute), backward[1])
        if last_compute_end < backward[1]:
            tail_nccl = _duration(
                _intersect_intervals(nccl, [(last_compute_end, backward[1])])
            )

    def milliseconds(value):
        return value / 1000.0

    def percent(value, total):
        return 100.0 * value / total if total > 0 else 0.0

    lines = [
        f"backward wall range:            {milliseconds(backward_duration):10.3f} ms",
        (
            f"NCCL union inside backward:     {milliseconds(nccl_time):10.3f} ms "
            f"({percent(nccl_time, backward_duration):6.2f}%)"
        ),
        (
            f"non-NCCL CUDA compute union:    {milliseconds(compute_time):10.3f} ms "
            f"({percent(compute_time, backward_duration):6.2f}%)"
        ),
        (
            f"GEMM/Tensor kernel union:       {milliseconds(gemm_time):10.3f} ms "
            f"({percent(gemm_time, backward_duration):6.2f}%)"
        ),
        (
            f"NCCL overlap with compute:      {milliseconds(nccl_compute_overlap):10.3f} ms "
            f"({percent(nccl_compute_overlap, nccl_time):6.2f}% of NCCL)"
        ),
        (
            f"NCCL overlap with GEMM:         {milliseconds(nccl_gemm_overlap):10.3f} ms "
            f"({percent(nccl_gemm_overlap, nccl_time):6.2f}% of NCCL)"
        ),
        (
            f"NCCL without CUDA compute:      {milliseconds(exposed_nccl):10.3f} ms "
            f"({percent(exposed_nccl, backward_duration):6.2f}% of backward)"
        ),
        (
            f"NCCL after last compute kernel: {milliseconds(tail_nccl):10.3f} ms "
            f"({percent(tail_nccl, backward_duration):6.2f}% of backward)"
        ),
    ]
    if not nccl:
        lines.append("result: no NCCL CUDA kernels were found inside backward.")
    elif exposed_nccl > 0.25 * backward_duration:
        lines.append("result: substantial NCCL time is not hidden by CUDA compute.")
    elif nccl_compute_overlap > 0.75 * nccl_time:
        lines.append("result: most NCCL activity overlaps CUDA compute.")
    else:
        lines.append("result: NCCL is partially overlapped.")
    return "\n".join(lines)
