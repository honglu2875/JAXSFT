#!/usr/bin/env python3
"""Compare batch-identical JAXSFT and Hugging Face training trajectories."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def _read_json(path: str | Path):
    return json.loads(Path(path).expanduser().resolve().read_text())


def _jax_events(path: str | Path) -> list[dict]:
    events = []
    for line in Path(path).expanduser().resolve().read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("event"), str):
            events.append(value)
    return events


def trajectory_stability(comparisons: list[dict], *, start_step: int) -> dict:
    """Summarize whether relative loss error grows after updates can affect loss."""

    window = [record for record in comparisons if int(record["step"]) >= start_step]
    if len(window) < 4:
        raise ValueError("trajectory stability requires at least four post-update measurements")
    xs = [float(record["step"]) for record in window]
    ys = [float(record["relative_error"]) for record in window]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    variance_x = sum((value - mean_x) ** 2 for value in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / variance_x
    midpoint = len(window) // 2
    early, late = ys[:midpoint], ys[midpoint:]
    early_mean = sum(early) / len(early)
    late_mean = sum(late) / len(late)
    return {
        "start_step": start_step,
        "measurements": len(window),
        "relative_error_slope_per_step": slope,
        "early_half": {
            "start_step": int(window[0]["step"]),
            "end_step": int(window[midpoint - 1]["step"]),
            "mean_relative_error": early_mean,
            "maximum_relative_error": max(early),
        },
        "late_half": {
            "start_step": int(window[midpoint]["step"]),
            "end_step": int(window[-1]["step"]),
            "mean_relative_error": late_mean,
            "maximum_relative_error": max(late),
        },
        "late_minus_early_mean_relative_error": late_mean - early_mean,
        "final_relative_error": ys[-1],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax-log", required=True)
    parser.add_argument("--jax-manifest", required=True)
    parser.add_argument("--hf-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--loss-atol", type=float, required=True)
    parser.add_argument("--loss-rtol", type=float, required=True)
    parser.add_argument("--max-relative-error-slope", type=float, required=True)
    parser.add_argument("--max-half-mean-growth", type=float, required=True)
    parser.add_argument("--stability-start-step", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = (
        args.loss_atol,
        args.loss_rtol,
        args.max_relative_error_slope,
        args.max_half_mean_growth,
    )
    if any(value < 0 for value in thresholds) or not all(
        math.isfinite(value) for value in thresholds
    ):
        raise ValueError("loss and stability tolerances must be finite and non-negative")
    events = _jax_events(args.jax_log)
    jax_steps = [event for event in events if event["event"] == "train_step"]
    completions = [event for event in events if event["event"] == "complete"]
    if len(completions) != 1 or not jax_steps:
        raise ValueError("JAX log must contain train_step events and exactly one complete event")
    jax_manifest = _read_json(args.jax_manifest)
    hf = _read_json(args.hf_result)
    if hf.get("status") != "complete":
        raise ValueError("Hugging Face trajectory is incomplete")
    recipe_identity = jax_manifest.get("recipe", {}).get("identity_hash")
    if recipe_identity is None:
        # Public recipe dictionaries intentionally carry the identity beside the manifest payload.
        initialized = [event for event in events if event["event"] == "initialized"]
        if len(initialized) != 1:
            raise ValueError("JAX log must contain one initialized event")
        recipe_identity = initialized[0].get("recipe")
    if hf.get("recipe_identity_sha256") != recipe_identity:
        raise ValueError("JAX and Hugging Face recipe identities differ")
    jax_tape = completions[0].get("batch_tape_identity_sha256")
    if not jax_tape:
        jax_tape = jax_manifest.get("batch_tape", {}).get("identity_sha256")
    if hf.get("batch_tape_identity_sha256") != jax_tape:
        raise ValueError("JAX and Hugging Face batch tape identities differ")
    hf_steps = hf.get("trajectory")
    if not isinstance(hf_steps, list) or len(jax_steps) != len(hf_steps):
        raise ValueError("JAX and Hugging Face trajectories have different lengths")

    comparisons = []
    for jax_metric, hf_metric in zip(jax_steps, hf_steps):
        step = int(jax_metric["step"])
        if int(hf_metric["step"]) != step:
            raise ValueError("trajectory step indices differ")
        if not math.isclose(
            float(jax_metric["loss_denominator"]),
            float(hf_metric["loss_denominator"]),
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise ValueError(f"selected loss denominator differs at step {step}")
        if int(jax_metric["input_tokens"]) != int(hf_metric["input_tokens"]):
            raise ValueError(f"input token count differs at step {step}")
        if not math.isclose(
            float(jax_metric["learning_rate"]),
            float(hf_metric["learning_rate"]),
            rel_tol=1e-6,
            abs_tol=1e-12,
        ):
            raise ValueError(f"update learning rate differs at step {step}")
        jax_loss, hf_loss = float(jax_metric["loss"]), float(hf_metric["loss"])
        absolute = abs(jax_loss - hf_loss)
        relative = absolute / max(abs(hf_loss), 1e-12)
        limit = args.loss_atol + args.loss_rtol * abs(hf_loss)
        comparisons.append(
            {
                "step": step,
                "jax_loss": jax_loss,
                "hf_loss": hf_loss,
                "absolute_error": absolute,
                "relative_error": relative,
                "allowed_error": limit,
                "within_tolerance": absolute <= limit,
                "jax_gradient_norm": float(jax_metric["gradient_norm"]),
                "hf_gradient_norm": hf_metric.get("gradient_norm"),
                "jax_selected_accuracy": float(jax_metric["selected_accuracy"]),
                "hf_selected_accuracy": float(hf_metric["selected_accuracy"]),
            }
        )
    if args.stability_start_step is None:
        nonzero_updates = [
            int(metric["step"])
            for metric in jax_steps
            if float(metric["learning_rate"]) > 0.0
        ]
        if not nonzero_updates:
            raise ValueError("trajectory has no nonzero parameter update")
        stability_start_step = nonzero_updates[0] + 1
    else:
        stability_start_step = args.stability_start_step
    stability = trajectory_stability(comparisons, start_step=stability_start_step)
    tolerance_passed = all(record["within_tolerance"] for record in comparisons)
    stability_passed = (
        stability["relative_error_slope_per_step"] <= args.max_relative_error_slope
        and stability["late_minus_early_mean_relative_error"] <= args.max_half_mean_growth
    )
    passed = tolerance_passed and stability_passed
    payload = {
        "schema_version": 2,
        "status": "passed" if passed else "failed",
        "contract": {
            "same_recipe_identity": recipe_identity,
            "same_batch_tape_identity": jax_tape,
            "exact_selected_denominators": True,
            "exact_input_token_counts": True,
            "matched_update_learning_rates": True,
            "loss_atol": args.loss_atol,
            "loss_rtol": args.loss_rtol,
            "max_relative_error_slope": args.max_relative_error_slope,
            "max_half_mean_growth": args.max_half_mean_growth,
            "precision_note": (
                "JAX and CPU numerical modes are recorded in the source run manifest and Hugging Face result. "
                "The stability window begins only after a nonzero update can affect the next measured loss."
            ),
        },
        "summary": {
            "steps": len(comparisons),
            "numerical_tolerance_passed": tolerance_passed,
            "trajectory_stability_passed": stability_passed,
            "maximum_absolute_loss_error": max(record["absolute_error"] for record in comparisons),
            "mean_absolute_loss_error": sum(record["absolute_error"] for record in comparisons) / len(comparisons),
            "maximum_relative_loss_error": max(record["relative_error"] for record in comparisons),
            "trajectory_stability": stability,
        },
        "jax_numerics": jax_manifest.get("numerics"),
        "jax_source": jax_manifest.get("source"),
        "hugging_face_software": hf.get("software"),
        "steps": comparisons,
    }
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite comparison result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(payload, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
