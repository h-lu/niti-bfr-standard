from __future__ import annotations

import json
import sqlite3
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .extract_braided import BraidedExtractionConfig
from .extract import ExtractionConfig
from .pipeline import (
    AnalysisResult,
    BRAIDED_METRIC_ALIAS_TO_KEY,
    BRAIDED_METRIC_KEY_TO_ALIAS,
    analyze_braided_video_quicklook,
    analyze_video,
    compute_braided_acceptance,
)
from .synth_braided import (
    BraidedSyntheticModel,
    BraidedSyntheticModelConfig,
    BraidedSyntheticRenderConfig,
    write_braided_synthetic_dataset,
)
from .synth import SyntheticRenderConfig, build_model_from_dict, generate_temperature_schedule, write_synthetic_dataset
from .temporal import RouteCConfig

ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "web"
TEMPLATES_DIR = WEB_ROOT / "templates"
DATA_ROOT = ROOT / "var" / "webapp"
RUNS_ROOT = DATA_ROOT / "runs"
DB_PATH = DATA_ROOT / "runs.db"
CONFIG_PATH = ROOT / "configs" / "minimal.yaml"

DATA_ROOT.mkdir(parents=True, exist_ok=True)
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="NiTi BFR Analysis Console")
app.mount("/files", StaticFiles(directory=str(DATA_ROOT)), name="files")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

SAMPLE_RUNS: dict[str, dict[str, str]] = {
    "wire_like_quicklook": {
        "label": "wire-like quicklook",
        "description": "直接使用仓库里的 data/wire-like.mp4 做 quicklook。",
        "preset": "wire_like",
        "requested_mode": "quicklook",
    },
    "synthetic_formal": {
        "label": "wire-like 合成 formal Af",
        "description": "现场生成一组仓库自带 synthetic demo，再用温度真值跑 formal Af。",
        "preset": "demo",
        "requested_mode": "formal_af",
    },
    "braided_synthetic_quicklook": {
        "label": "braided 合成 quicklook",
        "description": "现场生成一组 braided synthetic demo，不接温度文件，只看 braided 几何与 QC 快检。",
        "preset": "braided_demo",
        "requested_mode": "quicklook",
    },
    "braided_synthetic_formal": {
        "label": "braided 合成 formal Af",
        "description": "现场生成一组 braided synthetic demo，再用温度真值跑 formal Af。",
        "preset": "braided_demo",
        "requested_mode": "formal_af",
    },
}

PRESET_DISPLAY: dict[str, dict[str, str]] = {
    "wire_like": {
        "label": "wire-like",
        "family": "wire",
        "description": "细丝 / 针样对象，主 formal 量是 kappa_fit。",
    },
    "demo": {
        "label": "wire-like synthetic demo",
        "family": "wire",
        "description": "仓库内置的 wire-like 合成 demo。",
    },
    "braided_like": {
        "label": "braided-like",
        "family": "braided",
        "description": "编织网状器械对象，quicklook 看 A/B/C 与 body-only QC。",
    },
    "braided_demo": {
        "label": "braided synthetic demo",
        "family": "braided",
        "description": "仓库内置的 braided 合成 demo。",
    },
}

MODE_DISPLAY: dict[str, dict[str, str]] = {
    "quicklook": {
        "label": "quicklook",
        "description": "只给几何 / QC 快检，不自动等同于正式 Af 结论。",
    },
    "formal_af": {
        "label": "formal Af",
        "description": "需要温度同步，并且 formal gate 放行后才成立。",
    },
}


@app.middleware("http")
async def forwarded_prefix_middleware(request: Request, call_next):
    prefix = request.headers.get("x-forwarded-prefix", "").rstrip("/")
    if prefix:
        request.scope["root_path"] = prefix
    return await call_next(request)


@app.on_event("startup")
def _startup() -> None:
    _ensure_storage()


def _preset_label(preset: str | None) -> str:
    if preset is None:
        return "-"
    return PRESET_DISPLAY.get(preset, {}).get("label", preset)


def _preset_description(preset: str | None) -> str:
    if preset is None:
        return ""
    return PRESET_DISPLAY.get(preset, {}).get("description", "")


def _mode_label(mode: str | None) -> str:
    if mode is None:
        return "-"
    return MODE_DISPLAY.get(mode, {}).get("label", mode)


def _mode_description(mode: str | None) -> str:
    if mode is None:
        return ""
    return MODE_DISPLAY.get(mode, {}).get("description", "")


def _run_result_hint(run: sqlite3.Row | dict[str, Any]) -> str:
    requested_mode = run["requested_mode"]
    actual_mode = run["actual_mode"]
    gate_reason = run["formal_gate_reason"]
    status = run.get("status") if isinstance(run, dict) else run["status"]
    if status in {"queued", "running"}:
        return "任务执行中，等待实际结果。"
    if actual_mode == "formal_af":
        return "formal Af 已放行。"
    if requested_mode == "formal_af" and actual_mode == "quicklook":
        return f"formal Af 未放行，当前按 quicklook 展示（{gate_reason or 'gate_closed'}）。"
    if actual_mode == "quicklook":
        return "当前结果是 quicklook。"
    return gate_reason or "-"


templates.env.globals.update(
    preset_label=_preset_label,
    preset_description=_preset_description,
    mode_label=_mode_label,
    mode_description=_mode_description,
    run_result_hint=_run_result_hint,
)


@app.get("/")
def home(request: Request) -> Any:
    runs = _list_runs(limit=12)
    return templates.TemplateResponse(
        "index.html",
        {
            "runs": runs,
            "sample_runs": SAMPLE_RUNS,
            "preset_display": PRESET_DISPLAY,
            "mode_display": MODE_DISPLAY,
            "preset_label": _preset_label,
            "preset_description": _preset_description,
            "mode_label": _mode_label,
            "mode_description": _mode_description,
            "run_result_hint": _run_result_hint,
            "request": request,
        },
    )


@app.get("/history")
def history(request: Request) -> Any:
    runs = _list_runs(limit=200)
    return templates.TemplateResponse(
        "history.html",
        {
            "runs": runs,
            "preset_label": _preset_label,
            "mode_label": _mode_label,
            "run_result_hint": _run_result_hint,
            "request": request,
        },
    )


@app.post("/runs")
async def create_run(
    request: Request,
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    temperature_file: UploadFile | None = File(None),
    requested_mode: str = Form("quicklook"),
    preset: str = Form("wire_like"),
    run_name: str = Form(""),
) -> RedirectResponse:
    _ensure_storage()
    if requested_mode not in {"quicklook", "formal_af"}:
        raise HTTPException(status_code=400, detail="invalid requested_mode")
    if preset not in {"wire_like", "braided_like"}:
        raise HTTPException(status_code=400, detail="unsupported preset")
    if not video_file.filename:
        raise HTTPException(status_code=400, detail="video file is required")

    run_id = _new_run_id()
    run_dir = RUNS_ROOT / run_id
    inputs_dir = run_dir / "inputs"
    outputs_dir = run_dir / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    video_path = inputs_dir / _safe_filename(video_file.filename)
    await _save_upload(video_file, video_path)

    temperature_path: Path | None = None
    if temperature_file and temperature_file.filename:
        temperature_path = inputs_dir / _safe_filename(temperature_file.filename)
        await _save_upload(temperature_file, temperature_path)

    _insert_run(
        {
            "id": run_id,
            "created_at": _utc_now(),
            "status": "queued",
            "run_name": run_name.strip() or None,
            "preset": preset,
            "requested_mode": requested_mode,
            "actual_mode": None,
            "formal_metric_label": None,
            "formal_gate_reason": None,
            "af95_c": None,
            "aftan_c": None,
            "video_filename": video_path.name,
            "temperature_filename": temperature_path.name if temperature_path else None,
            "run_dir": str(run_dir),
            "error_text": None,
        }
    )
    background_tasks.add_task(_execute_run, run_id)
    return RedirectResponse(url=str(request.url_for("run_detail", run_id=run_id)), status_code=303)


@app.post("/sample-runs")
async def create_sample_run(
    request: Request,
    background_tasks: BackgroundTasks,
    sample_id: str = Form(...),
) -> RedirectResponse:
    _ensure_storage()
    if sample_id not in SAMPLE_RUNS:
        raise HTTPException(status_code=400, detail="invalid sample_id")

    sample = SAMPLE_RUNS[sample_id]
    run_id = _new_run_id()
    run_dir = RUNS_ROOT / run_id
    inputs_dir = run_dir / "inputs"
    outputs_dir = run_dir / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    sample_input = _prepare_sample_inputs(sample_id, inputs_dir)
    _insert_run(
        {
            "id": run_id,
            "created_at": _utc_now(),
            "status": "queued",
            "run_name": sample["label"],
            "preset": sample_input["preset"],
            "requested_mode": sample_input["requested_mode"],
            "actual_mode": None,
            "formal_metric_label": None,
            "formal_gate_reason": None,
            "af95_c": None,
            "aftan_c": None,
            "video_filename": sample_input["video_filename"],
            "temperature_filename": sample_input["temperature_filename"],
            "run_dir": str(run_dir),
            "error_text": None,
        }
    )
    background_tasks.add_task(_execute_run, run_id)
    return RedirectResponse(url=str(request.url_for("run_detail", run_id=run_id)), status_code=303)


@app.get("/runs/{run_id}")
def run_detail(request: Request, run_id: str) -> Any:
    run = _get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")

    summary = _load_summary(run_id)
    image_files = []
    download_files = []
    outputs_dir = RUNS_ROOT / run_id / "outputs"
    if outputs_dir.exists():
        for path in sorted(outputs_dir.iterdir()):
            rel = path.relative_to(DATA_ROOT).as_posix()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                image_files.append({"name": path.name, "url": str(request.url_for("files", path=rel))})
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".csv", ".json"}:
                download_files.append({"name": path.name, "url": str(request.url_for("files", path=rel))})

    return templates.TemplateResponse(
        "run_detail.html",
        {
            "request": request,
            "run": run,
            "summary": summary,
            "image_files": image_files,
            "download_files": download_files,
            "refresh": run["status"] in {"queued", "running"},
            "preset_label": _preset_label,
            "preset_description": _preset_description,
            "mode_label": _mode_label,
            "mode_description": _mode_description,
            "run_result_hint": _run_result_hint,
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _execute_run(run_id: str) -> None:
    run = _get_run(run_id)
    if run is None:
        return

    try:
        _update_run(run_id, {"status": "running", "error_text": None})
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        run_dir = Path(run["run_dir"])
        inputs_dir = run_dir / "inputs"
        outputs_dir = run_dir / "outputs"
        video_path = inputs_dir / run["video_filename"]
        temperature_path = None
        if run["requested_mode"] == "formal_af" and run["temperature_filename"]:
            temperature_path = inputs_dir / run["temperature_filename"]

        if run["preset"] in {"wire_like", "demo"}:
            extraction_cfg = _build_wire_extraction_config(config, run["preset"])
            route_c = RouteCConfig(**config["analysis"].get("wire_like_extraction", {}).get("route_c", {}))
            result = analyze_video(
                video_path,
                extraction=extraction_cfg,
                temperature_csv=temperature_path,
                route_c=route_c,
            )
        elif run["preset"] in {"braided_like", "braided_demo"}:
            extraction_cfg = _build_braided_extraction_config(config)
            result = analyze_braided_video_quicklook(
                video_path,
                extraction=extraction_cfg,
                temperature_csv=temperature_path,
            )
        else:
            raise RuntimeError(f"unsupported preset: {run['preset']}")

        result.series.to_csv(outputs_dir / "analysis.csv", index=False)
        summary = _build_summary(run, result)
        _write_summary(outputs_dir / "summary.json", summary)
        _write_plots(outputs_dir, result)

        _update_run(
            run_id,
            {
                "status": "completed",
                "actual_mode": result.mode,
                "formal_metric_label": _public_formal_metric_label(result),
                "formal_gate_reason": result.formal_gate_reason,
                "af95_c": result.af95_c,
                "aftan_c": result.aftan_c,
                "error_text": None,
            },
        )
    except Exception as exc:  # noqa: BLE001
        _update_run(run_id, {"status": "failed", "error_text": str(exc)})


def _public_formal_metric_label(result: AnalysisResult) -> str | None:
    return result.formal_metric_label if result.mode == "formal_af" else None


def _build_summary(run: sqlite3.Row, result: AnalysisResult) -> dict[str, Any]:
    series = result.series
    public_formal_metric_label = _public_formal_metric_label(result)
    summary: dict[str, Any] = {
        "run_id": run["id"],
        "preset": run["preset"],
        "requested_mode": run["requested_mode"],
        "actual_mode": result.mode,
        "formal_metric_label": public_formal_metric_label,
        "formal_gate_reason": result.formal_gate_reason,
        "af95_c": result.af95_c,
        "aftan_c": result.aftan_c,
        "frames": int(series["frame"].max()) + 1 if not series.empty else 0,
        "quality_median": float(series["quality"].median()) if "quality" in series else None,
        "video_filename": run["video_filename"],
        "temperature_filename": run["temperature_filename"],
    }
    if str(run["preset"]).startswith("braided"):
        summary["metric_aliases"] = BRAIDED_METRIC_ALIAS_TO_KEY
        summary["metric_display_labels"] = {
            key: f"{alias}:{key}" for alias, key in BRAIDED_METRIC_ALIAS_TO_KEY.items()
        }
        if public_formal_metric_label is not None:
            summary["formal_metric_alias"] = BRAIDED_METRIC_KEY_TO_ALIAS.get(public_formal_metric_label)
        if result.primary_metric_label is not None:
            summary["primary_metric_alias"] = BRAIDED_METRIC_KEY_TO_ALIAS.get(result.primary_metric_label)
        summary["acceptance"] = compute_braided_acceptance(series)
    if "temperature_c" in series.columns and series["temperature_c"].notna().any():
        summary["temperature_c_min"] = float(series["temperature_c"].min())
        summary["temperature_c_max"] = float(series["temperature_c"].max())
    if "length_axis_px" in series.columns:
        summary["length_axis_median_px"] = float(series["length_axis_px"].median())
    if "length_axis_body_bins_px" in series.columns:
        summary["length_axis_body_bins_median_px"] = float(series["length_axis_body_bins_px"].median())
    if "diameter_max_px" in series.columns:
        summary["diameter_max_median_px"] = float(series["diameter_max_px"].median())
    if "diameter_p95_px" in series.columns:
        summary["diameter_p95_median_px"] = float(series["diameter_p95_px"].median())
    if "diameter_mid_median_px" in series.columns:
        summary["diameter_mid_median_px"] = float(series["diameter_mid_median_px"].median())
    if "diameter_mid_p90_px" in series.columns:
        summary["diameter_mid_p90_median_px"] = float(series["diameter_mid_p90_px"].median())
    if "area_proj_px2" in series.columns:
        summary["area_proj_median_px2"] = float(series["area_proj_px2"].median())
    if "body_mask_area_px2" in series.columns:
        summary["body_mask_area_median_px2"] = float(series["body_mask_area_px2"].median())
    if "length_axis_disagreement_px" in series.columns:
        summary["length_axis_definition_gap_median_px"] = float(series["length_axis_disagreement_px"].median())
    if "body_mask_attachment_leak_fraction" in series.columns:
        summary["body_mask_attachment_leak_fraction_median"] = float(series["body_mask_attachment_leak_fraction"].median())
    if "endpoint_jump_px" in series.columns:
        summary["endpoint_jump_p95_px"] = float(series["endpoint_jump_px"].quantile(0.95))
    if "axis_peak_position_stability" in series.columns:
        summary["axis_peak_position_stability_p95"] = float(series["axis_peak_position_stability"].quantile(0.95))
    if result.metric_reports:
        summary["metric_reports"] = {
            key: {
                "af95_c": report.af95_c,
                "aftan_c": report.aftan_c,
                "fit_rmse": report.fit_rmse,
                "monotonic_violation_fraction": report.monotonic_violation_fraction,
            }
            for key, report in result.metric_reports.items()
        }
    return summary


def _write_plots(out_dir: Path, result: AnalysisResult) -> None:
    import matplotlib.pyplot as plt

    series = result.series
    if series.empty:
        return

    if {"time_sec", "x_route_a_px", "x_fit_px", "x_route_c_px"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["time_sec"], series["x_route_a_px"], label="route A x", linewidth=1.8)
        plt.plot(series["time_sec"], series["x_fit_px"], label="route B x_fit", linewidth=1.8)
        plt.plot(series["time_sec"], series["x_route_c_px"], label="route C x_route_c", linewidth=1.8)
        plt.xlabel("Time (s)")
        plt.ylabel("x (px)")
        plt.title("Quicklook x over time")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "quicklook_x_vs_time.png", dpi=160)
        plt.close(fig)

    if {"time_sec", "kappa_fit_px_inv", "kappa_route_c_px_inv"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["time_sec"], series["kappa_fit_px_inv"], label="kappa_fit", linewidth=1.8)
        plt.plot(series["time_sec"], series["kappa_route_c_px_inv"], label="kappa_route_c", linewidth=1.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Curvature (px^-1)")
        plt.title("Quicklook curvature over time")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "quicklook_kappa_vs_time.png", dpi=160)
        plt.close(fig)

    if {"time_sec", "length_env_px", "length_axis_px"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["time_sec"], series["length_env_px"], label="envelope length", linewidth=1.8)
        plt.plot(series["time_sec"], series["length_axis_px"], label="A:length_axis", linewidth=1.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Length (px)")
        plt.title("Braided lengths over time")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "quicklook_lengths_vs_time.png", dpi=160)
        plt.close(fig)

    if {"time_sec", "diameter_max_px"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["time_sec"], series["diameter_max_px"], label="B:diameter_max", linewidth=1.8)
        if "diameter_mid_median_px" in series.columns:
            plt.plot(series["time_sec"], series["diameter_mid_median_px"], label="mid-window median", linewidth=1.4)
        if "diameter_mid_p90_px" in series.columns:
            plt.plot(series["time_sec"], series["diameter_mid_p90_px"], label="mid-window p90 proxy", linewidth=1.4)
        plt.xlabel("Time (s)")
        plt.ylabel("Diameter (px)")
        plt.title("Braided D_max over time")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "quicklook_diameter_vs_time.png", dpi=160)
        plt.close(fig)

    if {"time_sec", "area_proj_px2"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["time_sec"], series["area_proj_px2"], label="C:area_proj", linewidth=1.8)
        if "area_proj_contour_width_integral_px2" in series.columns:
            plt.plot(series["time_sec"], series["area_proj_contour_width_integral_px2"], label="contour-width integral", linewidth=1.4)
        plt.xlabel("Time (s)")
        plt.ylabel("Projected area integral (px^2)")
        plt.title("Braided A_proj over time")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "quicklook_area_vs_time.png", dpi=160)
        plt.close(fig)

    if {"time_sec", "body_mask_area_px2", "length_axis_disagreement_px"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["time_sec"], series["body_mask_area_px2"], label="body mask area", linewidth=1.8)
        plt.plot(series["time_sec"], series["length_axis_disagreement_px"], label="axis definition gap", linewidth=1.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Body/QC metric")
        plt.title("Braided body-only QC over time")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "quicklook_body_qc_vs_time.png", dpi=160)
        plt.close(fig)

    if {"time_sec", "foreshortening_axis", "foreshortening_env"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["time_sec"], series["foreshortening_axis"], label="FS axis", linewidth=1.8)
        plt.plot(series["time_sec"], series["foreshortening_env"], label="FS env", linewidth=1.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Foreshortening")
        plt.title("Braided foreshortening over time")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "quicklook_foreshortening_vs_time.png", dpi=160)
        plt.close(fig)

    if "temperature_c" not in series.columns or not series["temperature_c"].notna().any():
        return

    if {"x_route_a_recovery", "x_fit_recovery", "x_route_c_recovery"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["temperature_c"], series["x_route_a_recovery"], label="route A recovery", linewidth=1.8)
        plt.plot(series["temperature_c"], series["x_fit_recovery"], label="route B recovery", linewidth=1.8)
        plt.plot(series["temperature_c"], series["x_route_c_recovery"], label="route C recovery", linewidth=1.8)
        if result.mode == "formal_af" and result.af95_c is not None:
            plt.axvline(result.af95_c, color="tab:green", linestyle="--", label=f"Af-95 {result.af95_c:.2f}C")
        if result.mode == "formal_af" and result.aftan_c is not None:
            plt.axvline(result.aftan_c, color="tab:red", linestyle="--", label=f"Af-tan {result.aftan_c:.2f}C")
        plt.xlabel("Temperature (C)")
        plt.ylabel("Recovery ratio")
        plt.title("Recovery over temperature")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "recovery_vs_temperature.png", dpi=160)
        plt.close(fig)

    if {"temperature_c", "kappa_fit_px_inv", "kappa_route_c_px_inv"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["temperature_c"], series["kappa_fit_px_inv"], label="kappa_fit", linewidth=1.8)
        plt.plot(series["temperature_c"], series["kappa_route_c_px_inv"], label="kappa_route_c", linewidth=1.8)
        plt.xlabel("Temperature (C)")
        plt.ylabel("Curvature (px^-1)")
        plt.title("Curvature over temperature")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "kappa_vs_temperature.png", dpi=160)
        plt.close(fig)

    if {"temperature_c", "length_axis_recovery", "length_env_recovery", "diameter_max_recovery", "area_proj_recovery"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["temperature_c"], series["length_axis_recovery"], label="A recovery", linewidth=1.8)
        plt.plot(series["temperature_c"], series["length_env_recovery"], label="env recovery", linewidth=1.8)
        plt.plot(series["temperature_c"], series["diameter_max_recovery"], label="B recovery", linewidth=1.8)
        plt.plot(series["temperature_c"], series["area_proj_recovery"], label="C recovery", linewidth=1.8)
        if result.mode == "formal_af" and result.af95_c is not None:
            plt.axvline(result.af95_c, color="tab:green", linestyle="--", label=f"Af-95 {result.af95_c:.2f}C")
        if result.mode == "formal_af" and result.aftan_c is not None:
            plt.axvline(result.aftan_c, color="tab:red", linestyle="--", label=f"Af-tan {result.aftan_c:.2f}C")
        plt.xlabel("Temperature (C)")
        plt.ylabel("Recovery ratio")
        plt.title("Braided recovery over temperature")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "braided_recovery_vs_temperature.png", dpi=160)
        plt.close(fig)

    if {"temperature_c", "length_env_px", "length_axis_px", "diameter_max_px"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["temperature_c"], series["length_env_px"], label="env length", linewidth=1.8)
        plt.plot(series["temperature_c"], series["length_axis_px"], label="A", linewidth=1.8)
        plt.plot(series["temperature_c"], series["diameter_max_px"], label="B", linewidth=1.8)
        plt.xlabel("Temperature (C)")
        plt.ylabel("Length / diameter (px)")
        plt.title("Braided A/B and envelope over temperature")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "braided_geometry_vs_temperature.png", dpi=160)
        plt.close(fig)

    if {"temperature_c", "body_mask_area_px2", "length_axis_disagreement_px", "diameter_peak_pos_norm"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["temperature_c"], series["body_mask_area_px2"], label="body mask area", linewidth=1.8)
        plt.plot(series["temperature_c"], series["length_axis_disagreement_px"], label="axis definition gap", linewidth=1.8)
        plt.plot(series["temperature_c"], series["diameter_peak_pos_norm"], label="peak pos norm", linewidth=1.8)
        plt.xlabel("Temperature (C)")
        plt.ylabel("Body/QC metric")
        plt.title("Braided body-only QC over temperature")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "braided_body_qc_vs_temperature.png", dpi=160)
        plt.close(fig)

def _build_wire_extraction_config(config: dict[str, Any], preset: str) -> ExtractionConfig:
    analysis = config["analysis"]
    raw = analysis.get("wire_like_extraction") if preset == "wire_like" else None
    if raw is None:
        raw = analysis.get("demo_extraction")
    if raw is None:
        raise RuntimeError("missing extraction preset")
    return ExtractionConfig(
        roi_xyxy=tuple(raw["roi_xyxy"]),
        blur_ksize=int(raw["blur_ksize"]),
        threshold_dark=int(raw["threshold_dark"]),
        open_kernel=int(raw["open_kernel"]),
        close_kernel=int(raw["close_kernel"]),
        route_a_tip_cluster_radius_px=float(raw.get("route_a_tip_cluster_radius_px", 6.0)),
        route_b_endpoint_extension_scale=float(raw.get("route_b_endpoint_extension_scale", 0.0)),
        route_b_cap_inset_scale=float(raw.get("route_b_cap_inset_scale", 0.08)),
        fit_bin_px=float(raw.get("fit_bin_px", 3.0)),
        fit_path_fraction=float(raw.get("fit_path_fraction", 0.72)),
        fit_path_fraction_min=float(raw.get("fit_path_fraction_min", 0.42)),
        fit_curvature_threshold_ratio=float(raw.get("fit_curvature_threshold_ratio", 0.28)),
        fit_margin_prefer_quadratic=float(raw.get("fit_margin_prefer_quadratic", 0.05)),
    )


def _build_braided_extraction_config(config: dict[str, Any]) -> BraidedExtractionConfig:
    raw = config["analysis"].get("braided_device_extraction")
    if raw is None:
        raise RuntimeError("missing braided extraction preset")
    return BraidedExtractionConfig(
        roi_xyxy=tuple(raw["roi_xyxy"]),
        blur_ksize=int(raw["blur_ksize"]),
        threshold_dark=int(raw["threshold_dark"]),
        open_kernel=int(raw["open_kernel"]),
        close_kernel=int(raw["close_kernel"]),
        min_component_area=int(raw.get("min_component_area", 200)),
        width_sampling_step_px=float(raw.get("width_sampling_step_px", 4.0)),
        taper_threshold_ratio=float(raw.get("taper_threshold_ratio", 0.25)),
        compaction_threshold_ratio=float(raw.get("compaction_threshold_ratio", 0.75)),
        qc_max_segments=int(raw.get("qc_max_segments", 15)),
        tube_radius_scale=float(raw.get("tube_radius_scale", 1.0)),
        body_min_halfwidth_px=float(raw.get("body_min_halfwidth_px", 3.0)),
        centerline_smooth_window=int(raw.get("centerline_smooth_window", 7)),
        diameter_peak_threshold_ratio=float(raw.get("diameter_peak_threshold_ratio", 0.95)),
        attachment_min_area_px2=int(raw.get("attachment_min_area_px2", 24)),
        attachment_orientation_mismatch_deg=float(raw.get("attachment_orientation_mismatch_deg", 35.0)),
        attachment_far_axis_distance_ratio=float(raw.get("attachment_far_axis_distance_ratio", 0.78)),
        attachment_short_axis_span_ratio=float(raw.get("attachment_short_axis_span_ratio", 0.18)),
        attachment_short_normal_span_ratio=float(raw.get("attachment_short_normal_span_ratio", 0.55)),
        attachment_prune_dilate_kernel=int(raw.get("attachment_prune_dilate_kernel", 1)),
        diameter_proxy_window_start_norm=float(raw.get("diameter_proxy_window_start_norm", 0.6)),
        diameter_proxy_window_end_norm=float(raw.get("diameter_proxy_window_end_norm", 0.8)),
    )


def _prepare_sample_inputs(sample_id: str, inputs_dir: Path) -> dict[str, str | None]:
    if sample_id == "wire_like_quicklook":
        src = ROOT / "data" / "wire-like.mp4"
        dst = inputs_dir / "wire-like.mp4"
        shutil.copy2(src, dst)
        return {
            "preset": "wire_like",
            "requested_mode": "quicklook",
            "video_filename": dst.name,
            "temperature_filename": None,
        }

    if sample_id == "synthetic_formal":
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
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
        write_synthetic_dataset(inputs_dir, model, render_cfg, schedule)
        return {
            "preset": "demo",
            "requested_mode": "formal_af",
            "video_filename": "synthetic.mp4",
            "temperature_filename": "temperature_truth.csv",
        }

    if sample_id in {"braided_synthetic_quicklook", "braided_synthetic_formal"}:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        synth_cfg = config["braided_synthetic"]
        render_cfg = BraidedSyntheticRenderConfig(
            image_width=int(synth_cfg["image_width"]),
            image_height=int(synth_cfg["image_height"]),
            fps=int(synth_cfg["fps"]),
            duration_sec=float(synth_cfg["duration_sec"]),
            background_gray=int(synth_cfg["render"]["background_gray"]),
            body_gray=int(synth_cfg["render"]["body_gray"]),
            wire_gray=int(synth_cfg["render"]["wire_gray"]),
            outline_gray=int(synth_cfg["render"]["outline_gray"]),
            braid_spacing_px=int(synth_cfg["render"]["braid_spacing_px"]),
            wire_thickness_px=int(synth_cfg["render"]["wire_thickness_px"]),
            outline_thickness_px=int(synth_cfg["render"]["outline_thickness_px"]),
            support_gray=int(synth_cfg["render"]["support_gray"]),
            support_length_px=int(synth_cfg["render"]["support_length_px"]),
            support_radius_px=int(synth_cfg["render"]["support_radius_px"]),
            tip_cap_gray=int(synth_cfg["render"]["tip_cap_gray"]),
            tip_cap_length_px=int(synth_cfg["render"]["tip_cap_length_px"]),
            tip_cap_radius_px=int(synth_cfg["render"]["tip_cap_radius_px"]),
            noise_sigma=float(synth_cfg["render"]["noise_sigma"]),
            blur_sigma=float(synth_cfg["render"]["blur_sigma"]),
        )
        model = BraidedSyntheticModel(
            BraidedSyntheticModelConfig(
                center_xy=tuple(float(v) for v in synth_cfg["center_xy"]),
                length_m_px=float(synth_cfg["length_m_px"]),
                length_a_px=float(synth_cfg["length_a_px"]),
                diameter_m_px=float(synth_cfg["diameter_m_px"]),
                diameter_a_px=float(synth_cfg["diameter_a_px"]),
                transition_temp_c=float(synth_cfg["transition_temp_c"]),
                transition_width_c=float(synth_cfg["transition_width_c"]),
                peak_shift_norm=float(synth_cfg["peak_shift_norm"]),
                left_profile_power=float(synth_cfg["left_profile_power"]),
                right_profile_power=float(synth_cfg["right_profile_power"]),
                bow_m_px=float(synth_cfg["bow_m_px"]),
                bow_a_px=float(synth_cfg["bow_a_px"]),
                axis_angle_m_deg=float(synth_cfg["axis_angle_m_deg"]),
                axis_angle_a_deg=float(synth_cfg["axis_angle_a_deg"]),
                center_shift_a_xy=tuple(float(v) for v in synth_cfg["center_shift_a_xy"]),
            )
        )
        schedule = generate_temperature_schedule(
            fps=render_cfg.fps,
            duration_sec=render_cfg.duration_sec,
            start_c=float(synth_cfg["temperature"]["start_c"]),
            end_c=float(synth_cfg["temperature"]["end_c"]),
        )
        braided_extraction_cfg = _build_braided_extraction_config(config)
        write_braided_synthetic_dataset(
            inputs_dir,
            model,
            render_cfg,
            schedule,
            taper_threshold_ratio=braided_extraction_cfg.taper_threshold_ratio,
            compaction_threshold_ratio=braided_extraction_cfg.compaction_threshold_ratio,
        )
        return {
            "preset": "braided_demo",
            "requested_mode": "quicklook" if sample_id == "braided_synthetic_quicklook" else "formal_af",
            "video_filename": "synthetic.mp4",
            "temperature_filename": None if sample_id == "braided_synthetic_quicklook" else "truth.csv",
        }

    raise ValueError(f"unsupported sample_id: {sample_id}")


def _ensure_storage() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    with _connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                run_name TEXT,
                preset TEXT NOT NULL,
                requested_mode TEXT NOT NULL,
                actual_mode TEXT,
                formal_metric_label TEXT,
                formal_gate_reason TEXT,
                af95_c REAL,
                aftan_c REAL,
                video_filename TEXT NOT NULL,
                temperature_filename TEXT,
                run_dir TEXT NOT NULL,
                error_text TEXT
            )
            """
        )


def _connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _insert_run(values: dict[str, Any]) -> None:
    columns = ", ".join(values.keys())
    placeholders = ", ".join(f":{key}" for key in values)
    with _connect_db() as conn:
        conn.execute(f"INSERT INTO runs ({columns}) VALUES ({placeholders})", values)
        conn.commit()


def _update_run(run_id: str, values: dict[str, Any]) -> None:
    assignments = ", ".join(f"{key} = :{key}" for key in values)
    payload = {**values, "id": run_id}
    with _connect_db() as conn:
        conn.execute(f"UPDATE runs SET {assignments} WHERE id = :id", payload)
        conn.commit()


def _get_run(run_id: str) -> sqlite3.Row | None:
    with _connect_db() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return row


def _list_runs(limit: int) -> list[sqlite3.Row]:
    with _connect_db() as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY datetime(created_at) DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return list(rows)


async def _save_upload(upload: UploadFile, path: Path) -> None:
    with path.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    await upload.close()


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_summary(run_id: str) -> dict[str, Any] | None:
    path = RUNS_ROOT / run_id / "outputs" / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_filename(name: str) -> str:
    base = Path(name).name.replace(" ", "_")
    return base or f"upload-{uuid.uuid4().hex[:8]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def main() -> None:
    import uvicorn

    uvicorn.run("niti_bfr.webapp:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
