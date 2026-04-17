from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from niti_bfr.extract import ExtractionConfig
from niti_bfr.pipeline import _metric_preference_score, analyze_video
from niti_bfr.synth import (
    SyntheticRenderConfig,
    build_model_from_dict,
    generate_temperature_schedule,
    write_synthetic_dataset,
)
from niti_bfr.temporal import RouteCConfig


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/minimal.yaml").read_text(encoding="utf-8"))
    synth_cfg = config["synthetic"]
    render_cfg = SyntheticRenderConfig(
        image_width=int(synth_cfg["image_width"]),
        image_height=int(synth_cfg["image_height"]),
        fps=int(synth_cfg["fps"]),
        duration_sec=float(synth_cfg["duration_sec"]),
        line_thickness_px=int(synth_cfg["line_thickness_px"]),
        background_gray=int(synth_cfg["render"]["background_gray"]),
        noise_sigma=float(synth_cfg["render"]["noise_sigma"]),
        blur_sigma=float(synth_cfg["render"]["blur_sigma"]),
    )
    model = build_model_from_dict(synth_cfg)
    schedule = generate_temperature_schedule(
        fps=render_cfg.fps,
        duration_sec=render_cfg.duration_sec,
        start_c=float(synth_cfg["temperature"]["start_c"]),
        end_c=float(synth_cfg["temperature"]["end_c"]),
    )
    out_dir = ROOT / "outputs/demo"
    truth = write_synthetic_dataset(out_dir, model, render_cfg, schedule)
    truth_metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))

    extraction_params = config["analysis"]["demo_extraction"]
    extraction_cfg = ExtractionConfig(
        roi_xyxy=tuple(extraction_params["roi_xyxy"]),
        blur_ksize=int(extraction_params["blur_ksize"]),
        threshold_dark=int(extraction_params["threshold_dark"]),
        open_kernel=int(extraction_params["open_kernel"]),
        close_kernel=int(extraction_params["close_kernel"]),
        route_a_tip_cluster_radius_px=float(extraction_params.get("route_a_tip_cluster_radius_px", 6.0)),
        fit_bin_px=float(extraction_params.get("fit_bin_px", 3.0)),
        fit_path_fraction=float(extraction_params.get("fit_path_fraction", 0.72)),
        fit_margin_prefer_quadratic=float(extraction_params.get("fit_margin_prefer_quadratic", 0.05)),
    )
    result = analyze_video(
        out_dir / "synthetic.mp4",
        extraction=extraction_cfg,
        temperature_csv=out_dir / "temperature_truth.csv",
        route_c=RouteCConfig(**extraction_params.get("route_c", {})),
    )
    result.series.to_csv(out_dir / "analysis.csv", index=False)

    plt.figure(figsize=(8, 4.8))
    plt.plot(truth["temperature_c"], truth["x_true_px"], label="x true", linewidth=2)
    plt.plot(result.series["temperature_c"], result.series["x_route_a_px"], label="route A x", alpha=0.75)
    plt.plot(result.series["temperature_c"], result.series["x_fit_px"], label="route B x", alpha=0.8)
    plt.plot(result.series["temperature_c"], result.series["x_route_c_px"], label="route C x", alpha=0.9)
    plt.xlabel("Temperature (C)")
    plt.ylabel("x (px)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "x_vs_temperature.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(result.series["temperature_c"], result.series["x_route_a_recovery"], label="R_route_a(T)", linewidth=2, alpha=0.8)
    plt.plot(result.series["temperature_c"], result.series["x_fit_recovery"], label="R_route_b(T)", linewidth=2)
    plt.plot(result.series["temperature_c"], result.series["x_route_c_recovery"], label="R_route_c(T)", linewidth=2)
    if result.af95_c is not None:
        label = f"{result.primary_metric_label or 'primary'} Af-95={result.af95_c:.2f}C"
        plt.axvline(result.af95_c, color="tab:green", linestyle="--", label=label)
    if result.aftan_c is not None:
        label = f"{result.primary_metric_label or 'primary'} Af-tan={result.aftan_c:.2f}C"
        plt.axvline(result.aftan_c, color="tab:red", linestyle="--", label=label)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Recovery ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "recovery_vs_temperature.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(result.series["temperature_c"], result.series["kappa_fit_px_inv"], label="kappa_fit", linewidth=2)
    plt.plot(result.series["temperature_c"], result.series["kappa_route_c_px_inv"], label="kappa_route_c", linewidth=2)
    plt.plot(truth["temperature_c"], truth["kappa_true_px_inv"], label="kappa true", linewidth=2, alpha=0.75)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Curvature (px^-1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "kappa_vs_temperature.png", dpi=160)
    plt.close()

    metric_reports = {}
    if result.metric_reports:
        for key, report in result.metric_reports.items():
            metric_reports[key] = {
                "label": report.label,
                "increasing": report.increasing,
                "af95_c": report.af95_c,
                "aftan_c": report.aftan_c,
                "fit_rmse": report.fit_rmse,
                "monotonic_violation_fraction": report.monotonic_violation_fraction,
                "dynamic_range": report.dynamic_range,
                "fit": report.fit.__dict__,
            }

    if result.metric_reports:
        preferred_metric = min(
            ((label, _metric_preference_score(report)) for label, report in result.metric_reports.items()),
            key=lambda item: item[1],
        )[0]
    else:
        preferred_metric = None

    metric_truth_comparison = None
    if result.metric_reports:
        metric_truth_comparison = {
            "x_route_a_vs_x_true": {
                "af95_error_c": result.metric_reports["x_route_a"].af95_c - truth_metrics["x_af95_c"],
                "aftan_error_c": result.metric_reports["x_route_a"].aftan_c - truth_metrics["x_aftan_c"],
            },
            "x_fit_vs_x_true": {
                "af95_error_c": result.metric_reports["x_fit"].af95_c - truth_metrics["x_af95_c"],
                "aftan_error_c": result.metric_reports["x_fit"].aftan_c - truth_metrics["x_aftan_c"],
            },
            "x_route_c_vs_x_true": {
                "af95_error_c": result.metric_reports["x_route_c"].af95_c - truth_metrics["x_af95_c"],
                "aftan_error_c": result.metric_reports["x_route_c"].aftan_c - truth_metrics["x_aftan_c"],
            },
            "kappa_fit_vs_kappa_true": {
                "af95_error_c": result.metric_reports["kappa_fit"].af95_c - truth_metrics["kappa_af95_c"],
                "aftan_error_c": result.metric_reports["kappa_fit"].aftan_c - truth_metrics["kappa_aftan_c"],
            },
            "kappa_route_c_vs_kappa_true": {
                "af95_error_c": result.metric_reports["kappa_route_c"].af95_c - truth_metrics["kappa_af95_c"],
                "aftan_error_c": result.metric_reports["kappa_route_c"].aftan_c - truth_metrics["kappa_aftan_c"],
            },
        }

    summary = {
        "demo_output": str(out_dir),
        "af95_c": result.af95_c,
        "aftan_c": result.aftan_c,
        "fit": result.fit.__dict__ if result.fit else None,
        "primary_metric_label": result.primary_metric_label,
        "preferred_metric_for_af": preferred_metric,
        "truth_metrics": truth_metrics,
        "metric_truth_comparison": metric_truth_comparison,
        "metric_reports": metric_reports,
    }
    (out_dir / "analysis_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
