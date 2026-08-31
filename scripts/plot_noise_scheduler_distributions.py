#!/usr/bin/env python3
"""Visualize the exact FastWAM training and inference noise schedules."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


PRESETS = {
    "noise": {
        "distribution": "shifted_uniform",
        "train_shift": 25.0,
        "display": "Noise (shift=25)",
        "color": "#d62728",
    },
    "baseline": {
        "distribution": "shifted_uniform",
        "train_shift": 5.0,
        "display": "Baseline (shift=5)",
        "color": "#1f77b4",
    },
    "middle": {
        "distribution": "logit_normal",
        "train_shift": 1.0,
        "display": "Middle (logit-normal)",
        "color": "#2ca02c",
    },
    "data": {
        "distribution": "shifted_uniform",
        "train_shift": 0.2,
        "display": "Data (shift=0.2)",
        "color": "#ff7f0e",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualizations/noise_scheduler_distributions"),
    )
    parser.add_argument("--num-samples", type=int, default=1_000_000)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def load_scheduler_class(repo_root: Path):
    module_path = (
        repo_root
        / "src"
        / "fastwam"
        / "models"
        / "wan22"
        / "schedulers"
        / "scheduler_continuous.py"
    )
    spec = importlib.util.spec_from_file_location(
        "fastwam_scheduler_continuous",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load scheduler module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.WanContinuousFlowMatchScheduler


def quantile_summary(values: torch.Tensor) -> dict:
    probabilities = torch.tensor(
        [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
        dtype=torch.float32,
    )
    quantiles = torch.quantile(values.float(), probabilities)
    return {
        "mean": float(values.float().mean()),
        "std": float(values.float().std()),
        **{
            f"q{round(probability.item() * 100):02d}": float(value)
            for probability, value in zip(probabilities, quantiles)
        },
    }


def write_inference_csv(path: Path, sigma: np.ndarray, delta: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "step",
                "timestep",
                "sigma_noise_coefficient",
                "data_coefficient",
                "delta",
                "absolute_delta",
            ]
        )
        for index, (sigma_value, delta_value) in enumerate(
            zip(sigma, delta),
            start=1,
        ):
            writer.writerow(
                [
                    index,
                    sigma_value * 1000.0,
                    sigma_value,
                    1.0 - sigma_value,
                    delta_value,
                    abs(delta_value),
                ]
            )


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive.")

    repo_root = Path(__file__).resolve().parents[1]
    scheduler_class = load_scheduler_class(repo_root)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    samples = {}
    schedulers = {}
    summary = {}
    for name, preset in PRESETS.items():
        scheduler = scheduler_class(
            num_train_timesteps=1000,
            shift=preset["train_shift"],
            sampling_distribution=preset["distribution"],
            logit_mean=0.0,
            logit_std=1.0,
        )
        sigma = scheduler.sample_training_t(
            batch_size=args.num_samples,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ) / 1000.0
        samples[name] = sigma
        schedulers[name] = scheduler
        summary[name] = {
            "training_distribution": preset["distribution"],
            "train_shift": preset["train_shift"],
            "sigma": quantile_summary(sigma),
            "mean_data_coefficient": float((1.0 - sigma).mean()),
        }

    # The ablations only change video_scheduler.train_shift. Both video and
    # action inference use infer_shift=5.0, so all four inference curves overlap.
    inference_scheduler = scheduler_class(
        num_train_timesteps=1000,
        shift=5.0,
    )
    timesteps, deltas = inference_scheduler.build_inference_schedule(
        num_inference_steps=args.num_inference_steps,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    inference_sigma = (timesteps / 1000.0).numpy()
    inference_delta = deltas.numpy()
    summary["inference"] = {
        "infer_shift": 5.0,
        "num_inference_steps": args.num_inference_steps,
        "all_four_presets_identical": True,
        "sigma": inference_sigma.tolist(),
        "delta": inference_delta.tolist(),
    }

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required; install it with `pip install matplotlib`."
        ) from exc

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.facecolor": "white",
        }
    )
    bins = np.linspace(0.0, 1.0, 301)
    centers = (bins[:-1] + bins[1:]) * 0.5

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    density_axis, cdf_axis, weight_axis, inference_axis = axes.flat
    for name, preset in PRESETS.items():
        sigma_np = samples[name].numpy()
        density, _ = np.histogram(sigma_np, bins=bins, density=True)
        density_axis.plot(
            centers,
            density,
            color=preset["color"],
            linewidth=2.0,
            label=preset["display"],
        )

        sorted_sigma = np.sort(sigma_np)
        cdf_indices = np.linspace(
            0,
            len(sorted_sigma) - 1,
            5000,
            dtype=np.int64,
        )
        cdf_axis.plot(
            sorted_sigma[cdf_indices],
            (cdf_indices + 1) / len(sorted_sigma),
            color=preset["color"],
            linewidth=2.0,
            label=preset["display"],
        )

        sigma_grid = torch.linspace(0.0, 1.0, 1001)
        weights = schedulers[name].training_weight(sigma_grid * 1000.0)
        weight_axis.plot(
            sigma_grid.numpy(),
            weights.numpy(),
            color=preset["color"],
            linewidth=2.0,
            label=preset["display"],
        )

    density_axis.set(
        title="Training timestep density",
        xlabel=r"$\sigma$ (noise coefficient)",
        ylabel=r"$p(\sigma)$",
        xlim=(0.0, 1.0),
    )
    density_axis.legend(frameon=False)
    cdf_axis.set(
        title="Training timestep CDF",
        xlabel=r"$\sigma$ (noise coefficient)",
        ylabel=r"$P(\Sigma \leq \sigma)$",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    weight_axis.set(
        title="Training loss weight",
        xlabel=r"$\sigma$ (noise coefficient)",
        ylabel="normalized weight",
        xlim=(0.0, 1.0),
    )

    inference_steps = np.arange(1, args.num_inference_steps + 1)
    inference_axis.plot(
        inference_steps,
        inference_sigma,
        marker="o",
        linewidth=2.0,
        color="#9467bd",
        label=r"noise coefficient $\sigma$",
    )
    inference_axis.plot(
        inference_steps,
        1.0 - inference_sigma,
        marker="s",
        linewidth=2.0,
        color="#8c564b",
        label=r"data coefficient $1-\sigma$",
    )
    inference_axis.set(
        title="Inference schedule (all four overlap, shift=5)",
        xlabel="Denoising step",
        ylabel="mixture coefficient",
        xticks=inference_steps,
        ylim=(-0.03, 1.03),
    )
    inference_axis.legend(frameon=False)
    fig.suptitle(
        "FastWAM video scheduler ablation: training distributions and inference",
        fontsize=14,
    )
    fig.savefig(output_dir / "noise_scheduler_distributions.png", dpi=args.dpi)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(
        inference_steps,
        inference_sigma,
        marker="o",
        linewidth=2.2,
        color="#9467bd",
    )
    axes[0].set(
        title="Inference noise level",
        xlabel="Denoising step",
        ylabel=r"$\sigma$ / timestep fraction",
        xticks=inference_steps,
        ylim=(-0.03, 1.03),
    )
    axes[1].bar(
        inference_steps,
        np.abs(inference_delta),
        color="#17becf",
        width=0.7,
    )
    axes[1].set(
        title="Euler step magnitude",
        xlabel="Denoising step",
        ylabel=r"$|\Delta \sigma|$",
        xticks=inference_steps,
    )
    fig.suptitle(
        "FastWAM 10-step inference schedule (infer_shift=5 for every preset)",
        fontsize=13,
    )
    fig.savefig(output_dir / "noise_scheduler_inference_schedule.png", dpi=args.dpi)
    plt.close(fig)

    write_inference_csv(
        output_dir / "noise_scheduler_inference_schedule.csv",
        inference_sigma,
        inference_delta,
    )
    (output_dir / "noise_scheduler_distribution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote scheduler visualizations to {output_dir}")
    for name in PRESETS:
        stats = summary[name]["sigma"]
        print(
            f"{name}: mean_sigma={stats['mean']:.4f} "
            f"median={stats['q50']:.4f} "
            f"q05={stats['q05']:.4f} q95={stats['q95']:.4f}"
        )
    print(f"inference sigma={inference_sigma.tolist()}")


if __name__ == "__main__":
    main()
