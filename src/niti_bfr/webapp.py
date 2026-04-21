from __future__ import annotations

from collections import Counter
import json
import sqlite3
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .extract_braided import BraidedExtractionConfig
from .export_contract import (
    ROUTE_ALIAS_ORDER,
    ROUTE_RESULTS_SCHEMA_VERSION,
    canonical_object_result_fields,
    route_results_dataframe,
    route_results_by_alias as canonical_route_results_by_alias,
)
from .extract import ExtractionConfig
from .metrics import af_95, af_tan
from .pipeline import (
    AnalysisResult,
    BRAIDED_FORMAL_CANDIDATES,
    BRAIDED_METRIC_ALIAS_TO_KEY,
    BRAIDED_METRIC_KEY_TO_ALIAS,
    WIRE_ROUTE_ALIAS_TO_KEY,
    analyze_braided_video_quicklook,
    analyze_video,
    compute_braided_acceptance,
)
from .process_debug_video import render_process_debug_video, transcode_video_for_browser
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
ASSETS_ROOT = WEB_ROOT / "assets"
DATA_ROOT = ROOT / "var" / "webapp"
RUNS_ROOT = DATA_ROOT / "runs"
PREVIEWS_ROOT = DATA_ROOT / "previews"
DB_PATH = DATA_ROOT / "runs.db"
CONFIG_PATH = ROOT / "configs" / "minimal.yaml"
PROJECT_OUTPUTS_ROOT = ROOT / "outputs"
ANNOTATED_VIDEO_FILENAME = "annotated_overview.mp4"
PROCESS_VIDEO_FILENAME = "analysis_process.mp4"
PROCESS_VIDEO_WEBM_FILENAME = "analysis_process.webm"

DATA_ROOT.mkdir(parents=True, exist_ok=True)
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
PREVIEWS_ROOT.mkdir(parents=True, exist_ok=True)
ASSETS_ROOT.mkdir(parents=True, exist_ok=True)

PATH_PREFIX_ALIASES = ("/niti",)

app = FastAPI(title="NiTi BFR 分析台")
app.mount("/assets", StaticFiles(directory=str(ASSETS_ROOT)), name="assets")
app.mount("/files", StaticFiles(directory=str(DATA_ROOT)), name="files")
for _prefix in PATH_PREFIX_ALIASES:
    app.mount(f"{_prefix}/assets", StaticFiles(directory=str(ASSETS_ROOT)), name=f"assets{_prefix}")
    app.mount(f"{_prefix}/files", StaticFiles(directory=str(DATA_ROOT)), name=f"files{_prefix}")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
_POSTPROCESS_LOCK = threading.Lock()
_POSTPROCESS_IN_FLIGHT: set[str] = set()
_SUMMARY_UNSET = object()

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
        "label": "细丝类 wire-like",
        "family": "wire",
        "description": "细丝 / 针样对象，长期保留 A / B / C 三条分析路线，当前主 formal 量是 kappa_fit。",
    },
    "demo": {
        "label": "细丝类合成示例",
        "family": "wire",
        "description": "仓库内置的细丝类合成示例数据。",
    },
    "braided_like": {
        "label": "编织类 braided-like",
        "family": "braided",
        "description": "编织网状器械对象，quicklook 查看 A / B / C 与 body-only 质检。",
    },
    "braided_demo": {
        "label": "编织类合成示例",
        "family": "braided",
        "description": "仓库内置的编织类合成示例数据。",
    },
}

MODE_DISPLAY: dict[str, dict[str, str]] = {
    "quicklook": {
        "label": "快速预览 quicklook",
        "description": "只给几何 / QC 快检，不自动等同于正式 Af 结论。",
    },
    "formal_af": {
        "label": "正式 Af",
        "description": "需要温度同步，并且 formal gate 放行后才成立。",
    },
}

ROUTE_STATUS_LABELS = {
    "quicklook_only": "仅快速预览",
    "formal_blocked": "formal 未放行",
    "provisional": "临时结果",
    "formal_passed": "formal 已放行",
    "missing": "暂无结果",
}
RUN_STATUS_LABELS = {
    "queued": "排队中",
    "running": "处理中",
    "completed": "已完成",
    "failed": "失败",
}
ROUTE_FORMAL_ROLE_LABELS = {
    "object_formal": "对象正式输出",
    "object_primary": "对象主推荐",
    "formal_candidate": "formal 候选",
    "route_result": "路线结果",
}
WORKER_OBJECT_LABELS = {
    "wire_like": "细丝对象",
    "demo": "细丝对象",
    "braided_like": "编织对象",
    "braided_demo": "编织对象",
}
WORKER_ROUTE_DETAILS = {
    "wire": {
        "A": {
            "name": "测量方式一",
            "title": "两端距离",
            "description": "看两端拉直了多少，最直观。",
        },
        "B": {
            "name": "测量方式二",
            "title": "整体弯曲程度",
            "description": "看主体整体变直了多少，通常最适合作为主结果。",
        },
        "C": {
            "name": "测量方式三",
            "title": "连续跟踪后的弯曲程度",
            "description": "结合前后画面一起判断，让结果更稳。",
        },
    },
    "braided": {
        "A": {
            "name": "测量方式一",
            "title": "主体长度",
            "description": "看主体沿主方向恢复了多少。",
        },
        "B": {
            "name": "测量方式二",
            "title": "主体宽度",
            "description": "看主体最宽处变化了多少。",
        },
        "C": {
            "name": "测量方式三",
            "title": "投影面积",
            "description": "看整体投影面积变化了多少。",
        },
    },
}
WORKER_ROUTE_STATUS_LABELS = {
    "formal_passed": "正式结果",
    "provisional": "当前汇报",
    "formal_blocked": "已计算",
    "quicklook_only": "趋势预览",
    "reportable": "正式结果",
    "reportable_with_warning": "当前汇报",
    "missing": "暂无结果",
}
WIRE_ROUTE_SPECS: dict[str, dict[str, Any]] = {
    "A": {
        "metric_key": WIRE_ROUTE_ALIAS_TO_KEY["A"],
        "display_label": "A:x_route_a",
        "metric_family": ("x_route_a",),
        "formal_candidate": False,
    },
    "B": {
        "metric_key": WIRE_ROUTE_ALIAS_TO_KEY["B"],
        "display_label": "B:kappa_fit",
        "metric_family": ("kappa_fit", "x_fit"),
        "formal_candidate": True,
    },
    "C": {
        "metric_key": WIRE_ROUTE_ALIAS_TO_KEY["C"],
        "display_label": "C:kappa_route_c",
        "metric_family": ("kappa_route_c", "x_route_c"),
        "formal_candidate": False,
    },
}
BRAIDED_ROUTE_SPECS: dict[str, dict[str, Any]] = {
    "A": {
        "metric_key": BRAIDED_METRIC_ALIAS_TO_KEY["A"],
        "display_label": "A:length_axis",
        "metric_family": ("length_axis", "length_env"),
        "formal_candidate": True,
    },
    "B": {
        "metric_key": BRAIDED_METRIC_ALIAS_TO_KEY["B"],
        "display_label": "B:diameter_max",
        "metric_family": ("diameter_max",),
        "formal_candidate": True,
    },
    "C": {
        "metric_key": BRAIDED_METRIC_ALIAS_TO_KEY["C"],
        "display_label": "C:area_proj",
        "metric_family": ("area_proj",),
        "formal_candidate": False,
    },
}
BENCHMARK_FAMILY_DISPLAY: dict[str, dict[str, Any]] = {
    "wire": {
        "label": "wire-like",
        "preset": "demo",
        "suite_path": PROJECT_OUTPUTS_ROOT / "wire_benchmark_suite" / "benchmark_summary.json",
    },
    "braided": {
        "label": "braided",
        "preset": "braided_demo",
        "suite_path": PROJECT_OUTPUTS_ROOT / "braided_benchmark_suite" / "benchmark_summary.json",
    },
}
WIRE_LEGACY_ROUTE_METRICS = {
    "A": ("x_route_a", "x_route_a_vs_x_true", "x_af95_c", "x_aftan_c"),
    "B": ("kappa_fit", "kappa_fit_vs_kappa_true", "kappa_af95_c", "kappa_aftan_c"),
    "C": ("kappa_route_c", "kappa_route_c_vs_kappa_true", "kappa_af95_c", "kappa_aftan_c"),
}
BRAIDED_SUITE_ROUTE_FIELDS = {
    "A": ("length_axis", "length_axis_af95_error_c", "length_axis_aftan_error_c"),
    "B": ("diameter_max", "diameter_af95_error_c", "diameter_aftan_error_c"),
    "C": ("area_proj", "area_af95_error_c", "area_aftan_error_c"),
}


@app.middleware("http")
async def forwarded_prefix_middleware(request: Request, call_next):
    prefix = request.headers.get("x-forwarded-prefix", "").rstrip("/")
    if not prefix:
        path = str(request.scope.get("path") or "")
        for candidate in PATH_PREFIX_ALIASES:
            if path == candidate or path.startswith(f"{candidate}/"):
                prefix = candidate
                break
    if prefix:
        request.scope["root_path"] = prefix
    return await call_next(request)


@app.on_event("startup")
def _startup() -> None:
    _ensure_storage()
    _cleanup_preview_cache()


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


def _run_status_label(status: str | None) -> str:
    if status is None:
        return "-"
    return RUN_STATUS_LABELS.get(status, status)


def _route_formal_role_label(role: str | None) -> str:
    if role is None:
        return "-"
    return ROUTE_FORMAL_ROLE_LABELS.get(role, role.replace("_", " "))


def _worker_object_label(preset: str | None) -> str:
    if preset is None:
        return "测量对象"
    return WORKER_OBJECT_LABELS.get(preset, "测量对象")


def _worker_route_detail(preset: str | None, alias: str | None, field: str) -> str:
    family = "braided" if _is_braided_preset(preset) else "wire"
    if alias is None:
        return "-"
    detail = WORKER_ROUTE_DETAILS.get(family, {}).get(alias, {})
    return str(detail.get(field) or "-")


def _worker_route_name(preset: str | None, alias: str | None) -> str:
    return _worker_route_detail(preset, alias, "name")


def _worker_route_title(preset: str | None, alias: str | None) -> str:
    return _worker_route_detail(preset, alias, "title")


def _worker_route_description(preset: str | None, alias: str | None) -> str:
    return _worker_route_detail(preset, alias, "description")


def _worker_route_status_label(status: str | None) -> str:
    if status is None:
        return "暂无结果"
    return WORKER_ROUTE_STATUS_LABELS.get(status, "暂无结果")


def _worker_gate_reason_label(reason: str | None) -> str:
    reason_text = str(reason or "").strip()
    if not reason_text:
        return "稳定性条件不足"
    lowered = reason_text.lower()
    if "temperature" in lowered:
        return "缺少可用温度信息"
    if "attachment" in lowered or "leak" in lowered:
        return "主体与附件分离不够稳定"
    if "centerline" in lowered:
        return "中心线提取不够稳定"
    if "endpoint" in lowered:
        return "端点识别不够稳定"
    if "dynamic_range" in lowered or "too_small" in lowered:
        return "变化幅度不够明显"
    if "insufficient" in lowered:
        return "有效数据不够"
    if "monotonic" in lowered:
        return "变化趋势不够稳定"
    return "稳定性条件不足"


def _worker_route_note(route: dict[str, Any] | None) -> str:
    route = route or {}
    if route.get("selected_as_formal"):
        return "本次正式结果采用这一种。"
    if route.get("selected_as_primary"):
        return "前端当前汇报这一种，formal gate 仅作调试提示。"
    status = route.get("reportability_status")
    if status == "provisional":
        return "前端当前汇报这一种，formal gate 仅作调试提示。"
    if status == "quicklook_only":
        return "当前没有温度结果，只看变化趋势。"
    if status == "formal_blocked":
        return f"已算出这一种的结果；{_worker_gate_reason_label(route.get('gate_reason'))}仅作算法调试提示。"
    if status == "formal_passed":
        return "这是一种可直接使用的结果。"
    return "当前暂无可展示说明。"


def _worker_result_summary(run: sqlite3.Row | dict[str, Any], summary: dict[str, Any] | None) -> dict[str, str]:
    status = run.get("status") if isinstance(run, dict) else run["status"]
    temperature_filename = (
        run.get("temperature_filename")
        if isinstance(run, dict)
        else run["temperature_filename"]
        if "temperature_filename" in run.keys()
        else None
    )
    actual_mode = (
        summary.get("actual_mode")
        if summary
        else run.get("actual_mode")
        if isinstance(run, dict)
        else run["actual_mode"]
    )
    gate_reason = (
        summary.get("formal_gate_reason")
        if summary
        else run.get("formal_gate_reason")
        if isinstance(run, dict)
        else run["formal_gate_reason"]
    )
    if status in {"queued", "running"}:
        return {
            "badge": "处理中",
            "headline": "系统正在分析，请稍等。",
            "description": "页面会自动刷新，结果出来后会直接显示在这里。",
        }
    if actual_mode == "formal_af":
        return {
            "badge": "可直接使用",
            "headline": "本次结果已经满足正式使用条件。",
            "description": "可以直接查看下面推荐的测量方式和结果温度。",
        }
    if temperature_filename:
        return {
            "badge": "已汇报结果",
            "headline": "这次已经算出可汇报的结果。",
            "description": f"formal gate 不阻塞前端显示；当前调试提示是{_worker_gate_reason_label(gate_reason)}。",
        }
    return {
        "badge": "只看趋势",
        "headline": "这次没有上传温度文件，所以只展示变化趋势。",
        "description": "如需温度结果，请下次同时上传视频和温度表。",
    }


def _worker_primary_route_label(summary: dict[str, Any] | None, preset: str | None) -> str:
    if not summary:
        return "-"
    alias = (
        summary.get("object_reported_route_alias")
        or summary.get("object_formal_route_alias")
        or summary.get("object_recommended_route_alias")
        or summary.get("recommended_route_alias")
        or summary.get("object_provisional_route_alias")
    )
    if not alias:
        return "-"
    return f"{_worker_route_name(preset, alias)}：{_worker_route_title(preset, alias)}"


def _worker_route_curve_label(preset: str | None, alias: str | None, kind: str) -> str:
    title = _worker_route_title(preset, alias)
    if title == "-":
        title = "测量方式"
    if kind == "temperature":
        return f"{title}温度曲线"
    if kind == "metric":
        return f"{title}变化图"
    return title


def _worker_output_label(filename: str, preset: str | None = None) -> str:
    route_curve_labels = {
        "route_a_metric_over_time.png": _worker_route_curve_label(preset, "A", "metric"),
        "route_b_metric_over_time.png": _worker_route_curve_label(preset, "B", "metric"),
        "route_c_metric_over_time.png": _worker_route_curve_label(preset, "C", "metric"),
        "route_a_recovery_vs_temperature.png": _worker_route_curve_label(preset, "A", "temperature"),
        "route_b_recovery_vs_temperature.png": _worker_route_curve_label(preset, "B", "temperature"),
        "route_c_recovery_vs_temperature.png": _worker_route_curve_label(preset, "C", "temperature"),
        "direction_metric_over_time.png": "方向法变化图",
        "direction_recovery_vs_temperature.png": "方向法温度曲线",
    }
    mapping = {
        "analysis_process.mp4": "分析过程视频",
        "analysis_process.webm": "分析过程视频",
        "analysis.csv": "每帧数据表",
        "route_results.csv": "三种测量方式结果表",
        "summary.json": "详细结果数据",
        "recovery_vs_temperature.png": "恢复温度曲线",
        "route_recovery_vs_temperature.png": "三种方式恢复温度曲线",
        "kappa_vs_temperature.png": "弯曲程度温度曲线",
        "quicklook_x_vs_time.png": "两端距离变化图",
        "quicklook_kappa_vs_time.png": "弯曲程度变化图",
    }
    mapping.update(route_curve_labels)
    return mapping.get(filename, filename)


def _plot_route_titles_for_series(series: pd.DataFrame) -> dict[str, str]:
    braided_columns = {
        "length_axis_px",
        "length_axis_recovery",
        "diameter_max_px",
        "diameter_max_recovery",
        "area_proj_px2",
        "area_proj_recovery",
    }
    preset = "braided_demo" if any(column in series.columns for column in braided_columns) else "demo"
    return {alias: _worker_route_title(preset, alias) for alias in ROUTE_ALIAS_ORDER}


def _curve_priority(filename: str, temperature_available: bool) -> tuple[int, str]:
    if temperature_available:
        order = {
            "route_recovery_vs_temperature.png": 0,
            "recovery_vs_temperature.png": 1,
            "kappa_vs_temperature.png": 2,
            "route_b_metric_over_time.png": 3,
            "route_a_metric_over_time.png": 4,
            "route_c_metric_over_time.png": 5,
            "direction_recovery_vs_temperature.png": 6,
            "direction_metric_over_time.png": 7,
        }
    else:
        order = {
            "route_b_metric_over_time.png": 0,
            "route_a_metric_over_time.png": 1,
            "route_c_metric_over_time.png": 2,
            "quicklook_x_vs_time.png": 3,
            "quicklook_kappa_vs_time.png": 4,
            "direction_metric_over_time.png": 5,
        }
    return order.get(filename, 99), filename


def _display_time(value: str | None) -> str:
    if not value:
        return "-"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if dt.tzinfo is not None:
        dt = dt.astimezone()
    return dt.strftime("%Y-%m-%d %H:%M")


def _ensure_video_poster(video_path: Path) -> Path | None:
    poster_path = video_path.with_name(f"{video_path.stem}_poster.jpg")
    if poster_path.exists():
        return poster_path
    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        return None
    if not cv2.imwrite(str(poster_path), frame):
        return None
    return poster_path


def _remove_video_with_poster(video_path: Path) -> None:
    for path in [video_path, video_path.with_name(f"{video_path.stem}_poster.jpg")]:
        try:
            path.unlink()
        except FileNotFoundError:
            continue


def _requested_flow_label(temperature_filename: str | None) -> str:
    if temperature_filename:
        return "正式 Af 提交流程"
    return "快速预览提交流程"


def _requested_mode_for_run(temperature_filename: str | None) -> str:
    return "formal_af" if temperature_filename else "quicklook"


def _run_result_hint(run: sqlite3.Row | dict[str, Any]) -> str:
    requested_mode = (
        run.get("requested_mode") if isinstance(run, dict) else run["requested_mode"] if "requested_mode" in run.keys() else None
    ) or _requested_mode_for_run(
        run.get("temperature_filename") if isinstance(run, dict) else run["temperature_filename"] if "temperature_filename" in run.keys() else None
    )
    actual_mode = run.get("actual_mode") if isinstance(run, dict) else run["actual_mode"]
    gate_reason = run.get("formal_gate_reason") if isinstance(run, dict) else run["formal_gate_reason"]
    reportability_status = run.get("reportability_status") if isinstance(run, dict) else run["reportability_status"] if "reportability_status" in run.keys() else None
    status = run.get("status") if isinstance(run, dict) else run["status"]
    if status in {"queued", "running"}:
        return "任务执行中，等待实际结果。"
    if actual_mode == "formal_af":
        return "formal Af 已放行。"
    if requested_mode == "formal_af" and reportability_status == "reportable_with_warning":
        return f"已汇报计算结果；formal gate 仅作调试提示（{gate_reason or 'gate_closed'}）。"
    if requested_mode == "formal_af" and actual_mode == "quicklook":
        return f"formal Af 未放行；当前以前端可见结果继续汇报（{gate_reason or 'gate_closed'}）。"
    if actual_mode == "quicklook":
        return "当前结果是快速预览 quicklook。"
    return gate_reason or "-"


def _is_braided_preset(preset: str | None) -> bool:
    return str(preset or "").startswith("braided")


def _route_specs_for_preset(preset: str | None) -> dict[str, dict[str, Any]]:
    return BRAIDED_ROUTE_SPECS if _is_braided_preset(preset) else WIRE_ROUTE_SPECS


def _route_status_label(status: str | None) -> str:
    return ROUTE_STATUS_LABELS.get(str(status or ""), str(status or "missing").replace("_", " "))


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _route_recovery_series_column(preset: str | None, alias: str) -> str | None:
    if _is_braided_preset(preset):
        return {
            "A": "length_axis_recovery",
            "B": "diameter_max_recovery",
            "C": "area_proj_recovery",
        }.get(alias)
    return {
        "A": "x_route_a_recovery",
        "B": "kappa_fit_recovery",
        "C": "kappa_route_c_recovery",
    }.get(alias)


def _smoothed_temperature_recovery(
    temp_c: Any,
    recovery: Any,
    *,
    window: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    temp_arr = np.asarray(temp_c, dtype=float)
    recovery_arr = np.asarray(recovery, dtype=float)
    valid = np.isfinite(temp_arr) & np.isfinite(recovery_arr)
    if np.count_nonzero(valid) < 4:
        return None

    temp_valid = temp_arr[valid]
    recovery_valid = recovery_arr[valid]
    order = np.argsort(temp_valid)
    temp_valid = temp_valid[order]
    recovery_valid = recovery_valid[order]

    smooth_window = min(int(window), len(recovery_valid))
    if smooth_window % 2 == 0:
        smooth_window -= 1
    if smooth_window >= 3:
        smoothed = (
            pd.Series(recovery_valid)
            .rolling(window=smooth_window, center=True, min_periods=1)
            .median()
            .to_numpy(dtype=float)
        )
    else:
        smooth_window = 1
        smoothed = recovery_valid
    return temp_valid, recovery_valid, smoothed, smooth_window


def _build_smoothed_route_results(preset: str | None, series: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if "temperature_c" not in series.columns:
        return {}

    route_smoothed: dict[str, dict[str, Any]] = {}
    for alias in ROUTE_ALIAS_ORDER:
        recovery_col = _route_recovery_series_column(preset, alias)
        if recovery_col is None or recovery_col not in series.columns:
            continue
        smoothed_payload = _smoothed_temperature_recovery(series["temperature_c"], series[recovery_col], window=7)
        if smoothed_payload is None:
            continue
        temp_valid, _raw_valid, smoothed, smooth_window = smoothed_payload
        route_smoothed[alias] = {
            "smoothed_af95_c": _coerce_float(af_95(temp_valid, smoothed)),
            "smoothed_aftan_c": _coerce_float(af_tan(temp_valid, smoothed)),
            "smoothing_method": "centered_rolling_median_temperature",
            "smoothing_window": smooth_window,
        }
    return route_smoothed


def _analysis_csv_path_for_run(run: sqlite3.Row | dict[str, Any]) -> Path:
    run_dir: str | None = None
    run_id: str | None = None
    if isinstance(run, dict):
        run_dir = run.get("run_dir")
        run_id = run.get("id")
    else:
        run_dir = run["run_dir"] if "run_dir" in run.keys() else None
        run_id = run["id"] if "id" in run.keys() else None
    if run_dir:
        return Path(str(run_dir)) / "outputs" / "analysis.csv"
    return RUNS_ROOT / str(run_id or "") / "outputs" / "analysis.csv"


def _backfill_smoothed_summary_fields(
    run: sqlite3.Row | dict[str, Any],
    prepared: dict[str, Any],
) -> dict[str, Any]:
    route_results = prepared.get("route_results") or []
    has_route_smoothed = any(
        entry.get("smoothed_af95_c") is not None or entry.get("smoothed_aftan_c") is not None for entry in route_results
    )
    has_object_smoothed = (
        prepared.get("object_smoothed_af95_c") is not None or prepared.get("object_smoothed_aftan_c") is not None
    )
    if has_route_smoothed and has_object_smoothed:
        return prepared

    analysis_csv = _analysis_csv_path_for_run(run)
    if not analysis_csv.exists():
        return prepared
    try:
        series = pd.read_csv(analysis_csv)
    except Exception:  # noqa: BLE001
        return prepared

    preset = prepared.get("preset") or (run.get("preset") if isinstance(run, dict) else run["preset"])
    smoothed_route_results = _build_smoothed_route_results(preset, series)
    if not smoothed_route_results:
        return prepared
    prepared["_smoothed_backfilled"] = True

    updated_route_results: list[dict[str, Any]] = []
    for route in route_results:
        route_copy = dict(route)
        route_copy.update(smoothed_route_results.get(route_copy.get("alias"), {}))
        updated_route_results.append(route_copy)
    prepared["route_results"] = updated_route_results
    prepared["route_results_by_alias"] = canonical_route_results_by_alias(updated_route_results)
    prepared.setdefault("route_smoothing_method", "centered_rolling_median_temperature")
    prepared.setdefault("route_smoothing_window", 7)

    if not has_object_smoothed:
        preferred_metric_key = (
            prepared.get("formal_metric_label")
            or prepared.get("object_formal_metric_key")
            or prepared.get("provisional_metric_label")
            or prepared.get("object_provisional_metric_key")
        )
        preferred_alias = _route_alias_from_metric_key(preset, preferred_metric_key)
        preferred_smoothed = smoothed_route_results.get(preferred_alias or "")
        if preferred_smoothed is not None:
            prepared["object_smoothed_metric_key"] = preferred_metric_key
            prepared["object_smoothed_route_alias"] = preferred_alias
            prepared["object_smoothed_af95_c"] = preferred_smoothed.get("smoothed_af95_c")
            prepared["object_smoothed_aftan_c"] = preferred_smoothed.get("smoothed_aftan_c")

    summary_path = _summary_path_for_run(run)
    existing_summary = _safe_read_json(summary_path)
    if isinstance(existing_summary, dict):
        existing_summary["route_results"] = updated_route_results
        existing_summary["route_results_by_alias"] = prepared["route_results_by_alias"]
        existing_summary["route_smoothing_method"] = prepared.get("route_smoothing_method")
        existing_summary["route_smoothing_window"] = prepared.get("route_smoothing_window")
        if prepared.get("object_smoothed_metric_key") is not None:
            existing_summary["object_smoothed_metric_key"] = prepared.get("object_smoothed_metric_key")
        if prepared.get("object_smoothed_route_alias") is not None:
            existing_summary["object_smoothed_route_alias"] = prepared.get("object_smoothed_route_alias")
        if prepared.get("object_smoothed_af95_c") is not None:
            existing_summary["object_smoothed_af95_c"] = prepared.get("object_smoothed_af95_c")
        if prepared.get("object_smoothed_aftan_c") is not None:
            existing_summary["object_smoothed_aftan_c"] = prepared.get("object_smoothed_aftan_c")
        _write_summary(summary_path, existing_summary)
        prepared.pop("_smoothed_backfilled", None)
    return prepared


def _result_for_display_plots(
    run: sqlite3.Row | dict[str, Any],
    prepared: dict[str, Any],
    series: pd.DataFrame,
) -> AnalysisResult:
    preset = prepared.get("preset") or (run.get("preset") if isinstance(run, dict) else run["preset"])
    _ = preset  # keep parity with call sites and future route-specific branching
    return AnalysisResult(
        series=series,
        fit=None,
        af95_c=_coerce_float(prepared.get("af95_c")),
        aftan_c=_coerce_float(prepared.get("aftan_c")),
        metric_reports=None,
        primary_metric_label=prepared.get("primary_metric_label"),
        mode=str(prepared.get("actual_mode") or (run.get("actual_mode") if isinstance(run, dict) else run["actual_mode"]) or "quicklook"),
        formal_metric_label=prepared.get("formal_metric_label"),
        formal_gate_reason=prepared.get("formal_gate_reason"),
        provisional_metric_label=prepared.get("provisional_metric_label"),
        provisional_af95_c=_coerce_float(prepared.get("provisional_af95_c")),
        provisional_aftan_c=_coerce_float(prepared.get("provisional_aftan_c")),
        reportability_status=str(prepared.get("reportability_status") or "quicklook_only"),
        warning_codes=list(prepared.get("warning_codes") or []),
        acceptance_profile=prepared.get("acceptance_profile"),
        route_results=list(prepared.get("route_results") or []),
        direction_result=prepared.get("direction_result"),
        formal_candidate_gates=prepared.get("formal_candidate_gates"),
        input_fps=_coerce_float(prepared.get("input_fps")),
        original_frame_count=prepared.get("original_frame_count"),
        analyzed_frame_count=prepared.get("analyzed_frame_count"),
        frame_stride=int(prepared.get("frame_stride") or 1),
    )


def _refresh_plot_outputs_for_display(
    run: sqlite3.Row | dict[str, Any],
    prepared: dict[str, Any] | None,
) -> None:
    if prepared is None:
        return
    status = (run.get("status") if isinstance(run, dict) else run["status"]) or "completed"
    if status != "completed":
        return
    if not _postprocess_needed(run, prepared):
        return
    run_id = run.get("id") if isinstance(run, dict) else run["id"]
    force_inline = _should_inline_recover_postprocess(run_id, prepared)
    _schedule_postprocess(run_id, run=run, prepared=prepared, force_inline=force_inline)


def _outputs_dir_for_run(run: sqlite3.Row | dict[str, Any]) -> Path:
    return _analysis_csv_path_for_run(run).parent


def _inputs_dir_for_run(run: sqlite3.Row | dict[str, Any]) -> Path:
    run_dir = run.get("run_dir") if isinstance(run, dict) else run["run_dir"]
    return Path(str(run_dir)) / "inputs"


def _summary_path_for_run(run: sqlite3.Row | dict[str, Any] | str) -> Path:
    if isinstance(run, str):
        return RUNS_ROOT / run / "outputs" / "summary.json"
    run_dir = run.get("run_dir") if isinstance(run, dict) else run["run_dir"]
    return Path(str(run_dir)) / "outputs" / "summary.json"


def _process_video_ready(outputs_dir: Path, prepared: dict[str, Any]) -> bool:
    filename = str(prepared.get("process_video_filename") or "").strip()
    if not filename:
        return False
    return (outputs_dir / filename).exists()


def _direction_metric_enabled_for_outputs(
    run: sqlite3.Row | dict[str, Any],
    prepared: dict[str, Any],
) -> bool:
    run_direction_enabled = (
        bool(run.get("direction_metric_enabled"))
        if isinstance(run, dict)
        else bool(run["direction_metric_enabled"]) if "direction_metric_enabled" in run.keys() else False
    )
    direction_result = prepared.get("direction_result")
    if isinstance(direction_result, dict):
        return bool(direction_result.get("enabled")) or run_direction_enabled
    return bool(direction_result) or run_direction_enabled


def _expected_plot_outputs(
    run: sqlite3.Row | dict[str, Any],
    prepared: dict[str, Any],
) -> set[str]:
    preset = str(prepared.get("preset") or (run.get("preset") if isinstance(run, dict) else run["preset"]) or "")
    temperature_available = bool(
        run.get("temperature_filename") if isinstance(run, dict) else run["temperature_filename"]
    ) or (
        prepared.get("temperature_c_min") is not None
        or prepared.get("temperature_c_max") is not None
    )
    expected = {
        "route_a_metric_over_time.png",
        "route_b_metric_over_time.png",
        "route_c_metric_over_time.png",
    }
    if preset.startswith("braided"):
        expected.update(
            {
                "quicklook_lengths_vs_time.png",
                "quicklook_diameter_vs_time.png",
                "quicklook_area_vs_time.png",
                "quicklook_body_qc_vs_time.png",
                "quicklook_foreshortening_vs_time.png",
            }
        )
    else:
        expected.update({"quicklook_x_vs_time.png", "quicklook_kappa_vs_time.png"})

    direction_enabled = _direction_metric_enabled_for_outputs(run, prepared)
    if direction_enabled:
        expected.add("direction_metric_over_time.png")

    if temperature_available:
        expected.update(
            {
                "route_a_recovery_vs_temperature.png",
                "route_b_recovery_vs_temperature.png",
                "route_c_recovery_vs_temperature.png",
                "route_recovery_vs_temperature.png",
            }
        )
        if preset.startswith("braided"):
            expected.update(
                {
                    "braided_recovery_vs_temperature.png",
                    "braided_geometry_vs_temperature.png",
                    "braided_body_qc_vs_temperature.png",
                }
            )
        else:
            expected.update({"recovery_vs_temperature.png", "kappa_vs_temperature.png"})
        if direction_enabled:
            expected.add("direction_recovery_vs_temperature.png")
    return expected


def _plot_outputs_complete(
    run: sqlite3.Row | dict[str, Any],
    prepared: dict[str, Any],
) -> bool:
    outputs_dir = _outputs_dir_for_run(run)
    if not outputs_dir.exists():
        return False
    expected = _expected_plot_outputs(run, prepared)
    return all((outputs_dir / filename).exists() for filename in expected)


def _postprocess_needed(
    run: sqlite3.Row | dict[str, Any],
    prepared: dict[str, Any] | None,
) -> bool:
    if prepared is None:
        return False
    return bool(
        prepared.get("asset_generation_status") in {"pending", "running", "failed"}
        or prepared.get("_smoothed_backfilled")
        or not _plot_outputs_complete(run, prepared)
        or not _process_video_ready(_outputs_dir_for_run(run), prepared)
    )


def _asset_generation_active(summary: dict[str, Any] | None) -> bool:
    if not isinstance(summary, dict):
        return False
    return str(summary.get("asset_generation_status") or "") in {"pending", "running"}


def _postprocess_in_flight(run_id: str) -> bool:
    with _POSTPROCESS_LOCK:
        return run_id in _POSTPROCESS_IN_FLIGHT


def _should_inline_recover_postprocess(run_id: str, prepared: dict[str, Any] | None) -> bool:
    if not isinstance(prepared, dict):
        return False
    if str(prepared.get("asset_generation_status") or "") != "running":
        return False
    return not _postprocess_in_flight(run_id)


def _postprocess_fallback_error(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__


def _postprocess_final_status(process_video_error: str | None, plot_error: str | None) -> tuple[str, str | None]:
    if process_video_error is None and plot_error is None:
        return "completed", None
    error_text = "; ".join(part for part in [process_video_error, plot_error] if part) or None
    return "failed", error_text


def _schedule_postprocess(
    run_id: str,
    *,
    run: sqlite3.Row | dict[str, Any] | None = None,
    prepared: dict[str, Any] | None = None,
    result: AnalysisResult | None = None,
    extraction_cfg: ExtractionConfig | BraidedExtractionConfig | None = None,
    video_path: Path | None = None,
    force_inline: bool = False,
) -> bool:
    with _POSTPROCESS_LOCK:
        if run_id in _POSTPROCESS_IN_FLIGHT:
            return False
        _POSTPROCESS_IN_FLIGHT.add(run_id)
    try:
        run_inline = force_inline or result is not None or bool(prepared and prepared.get("_smoothed_backfilled"))
        if run_inline:
            _run_postprocess_task(
                run_id=run_id,
                run=dict(run) if run is not None else None,
                prepared=dict(prepared) if prepared is not None else None,
                result=result,
                extraction_cfg=extraction_cfg,
                video_path=Path(video_path) if video_path is not None else None,
            )
            return True
        worker = threading.Thread(
            target=_run_postprocess_task,
            kwargs={
                "run_id": run_id,
                "run": dict(run) if run is not None else None,
                "prepared": dict(prepared) if prepared is not None else None,
                "result": result,
                "extraction_cfg": extraction_cfg,
                "video_path": Path(video_path) if video_path is not None else None,
            },
            daemon=True,
            name=f"niti-bfr-postprocess-{run_id[:12]}",
        )
        worker.start()
        return True
    except Exception:  # noqa: BLE001
        with _POSTPROCESS_LOCK:
            _POSTPROCESS_IN_FLIGHT.discard(run_id)
        return False


def _run_postprocess_task(
    *,
    run_id: str,
    run: dict[str, Any] | None,
    prepared: dict[str, Any] | None,
    result: AnalysisResult | None,
    extraction_cfg: ExtractionConfig | BraidedExtractionConfig | None,
    video_path: Path | None,
) -> None:
    try:
        run_payload = dict(run) if run is not None else None
        if run_payload is None:
            run_row = _get_run(run_id)
            if run_row is None:
                return
            run_payload = dict(run_row)

        outputs_dir = _outputs_dir_for_run(run_payload)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        analysis_csv = outputs_dir / "analysis.csv"
        if not analysis_csv.exists():
            return

        prepared_summary = dict(prepared) if prepared is not None else _prepare_summary_for_display(run_payload, _load_summary(run_id))
        if prepared_summary is None:
            return

        _update_summary_fields(
            run_payload,
            asset_generation_status="running",
            asset_generation_error=None,
        )

        analysis_result = result
        if analysis_result is None:
            try:
                series = pd.read_csv(analysis_csv)
            except Exception:  # noqa: BLE001
                return
            analysis_result = _result_for_display_plots(run_payload, prepared_summary, series)

        if extraction_cfg is None:
            config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
            preset = run_payload["preset"]
            if preset in {"wire_like", "demo"}:
                extraction_cfg = _build_wire_extraction_config(config, preset)
            elif preset in {"braided_like", "braided_demo"}:
                extraction_cfg = _build_braided_extraction_config(config)
            else:
                return

        resolved_video_path = Path(video_path) if video_path is not None else _inputs_dir_for_run(run_payload) / run_payload["video_filename"]
        process_video_error = _generate_process_video_asset(
            run_payload,
            result=analysis_result,
            extraction_cfg=extraction_cfg,
            video_path=resolved_video_path,
        )
        plot_error = _generate_plot_assets(
            run_payload,
            result=analysis_result,
            prepared=prepared_summary,
            force=bool(prepared_summary.get("_smoothed_backfilled")),
        )
        asset_generation_status, asset_generation_error = _postprocess_final_status(process_video_error, plot_error)
        _update_summary_fields(
            run_payload,
            asset_generation_status=asset_generation_status,
            asset_generation_error=asset_generation_error,
        )
    except Exception as exc:  # noqa: BLE001
        _update_summary_fields(
            run_id,
            asset_generation_status="failed",
            asset_generation_error=_postprocess_fallback_error(exc),
        )
    finally:
        with _POSTPROCESS_LOCK:
            _POSTPROCESS_IN_FLIGHT.discard(run_id)


def _update_summary_fields(
    run: sqlite3.Row | dict[str, Any] | str,
    *,
    process_video_filename: str | None | object = _SUMMARY_UNSET,
    process_video_error: str | None | object = _SUMMARY_UNSET,
    asset_generation_status: str | None | object = _SUMMARY_UNSET,
    asset_generation_error: str | None | object = _SUMMARY_UNSET,
) -> None:
    path = _summary_path_for_run(run)
    if not path.exists():
        return
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return
    if process_video_filename is not _SUMMARY_UNSET:
        summary["process_video_filename"] = process_video_filename
    if process_video_error is not _SUMMARY_UNSET:
        if process_video_error in {None, ""}:
            summary.pop("process_video_error", None)
        else:
            summary["process_video_error"] = process_video_error
    if asset_generation_status is not _SUMMARY_UNSET:
        summary["asset_generation_status"] = asset_generation_status
    if asset_generation_error is not _SUMMARY_UNSET:
        if asset_generation_error in {None, ""}:
            summary.pop("asset_generation_error", None)
        else:
            summary["asset_generation_error"] = asset_generation_error
    _write_summary(path, summary)


def _generate_process_video_asset(
    run: sqlite3.Row | dict[str, Any],
    *,
    result: AnalysisResult,
    extraction_cfg: ExtractionConfig | BraidedExtractionConfig,
    video_path: Path,
) -> str | None:
    outputs_dir = _outputs_dir_for_run(run)
    process_video_path = outputs_dir / PROCESS_VIDEO_FILENAME
    process_video_webm_path = outputs_dir / PROCESS_VIDEO_WEBM_FILENAME
    run_id = run.get("id") if isinstance(run, dict) else run["id"]
    summary = _load_summary(run_id) or {}
    existing_name = str(summary.get("process_video_filename") or "").strip()
    if existing_name and (outputs_dir / existing_name).exists():
        return None

    _remove_video_with_poster(process_video_path)
    _remove_video_with_poster(process_video_webm_path)
    try:
        render_result = render_process_debug_video(
            video_path=video_path,
            output_path=process_video_path,
            extraction=extraction_cfg,
            result=result,
            object_type=run.get("preset") if isinstance(run, dict) else run["preset"],
            output_fps=max((result.input_fps or 1.0) / max(result.frame_stride, 1), 1.0),
        )
        actual_output_path: Path | None
        if isinstance(render_result, Path):
            actual_output_path = render_result
        elif process_video_path.exists():
            actual_output_path = process_video_path
        elif process_video_webm_path.exists():
            actual_output_path = process_video_webm_path
        else:
            actual_output_path = None
        _update_summary_fields(
            run,
            process_video_filename=actual_output_path.name if actual_output_path is not None else None,
            process_video_error=None,
        )
        return None
    except Exception as exc:  # noqa: BLE001
        error_text = str(exc)
        _update_summary_fields(
            run,
            process_video_filename=None,
            process_video_error=error_text,
        )
        return error_text


def _generate_plot_assets(
    run: sqlite3.Row | dict[str, Any],
    *,
    result: AnalysisResult,
    prepared: dict[str, Any] | None = None,
    force: bool = False,
) -> str | None:
    outputs_dir = _outputs_dir_for_run(run)
    if prepared is not None and not force and _plot_outputs_complete(run, prepared):
        return None
    try:
        _write_plots(outputs_dir, result)
    except Exception as exc:  # noqa: BLE001
        return str(exc)
    return None


def _coerce_warning_codes(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item not in {None, ""}]
    if value == "":
        return []
    return [str(value)]


def _metric_report_field(report: Any, field: str) -> Any:
    if report is None:
        return None
    if isinstance(report, dict):
        return report.get(field)
    return getattr(report, field, None)


def _route_alias_from_metric_key(preset: str | None, metric_key: str | None) -> str | None:
    if metric_key is None:
        return None
    for alias, spec in _route_specs_for_preset(preset).items():
        if metric_key in spec["metric_family"]:
            return alias
    return None


def _route_metric_report(metric_reports: dict[str, Any] | None, spec: dict[str, Any]) -> Any:
    if not metric_reports:
        return None
    for metric_key in spec["metric_family"]:
        if metric_key in metric_reports:
            return metric_reports[metric_key]
    return None


def _temperature_available_for_summary(summary_like: dict[str, Any]) -> bool:
    if summary_like.get("temperature_filename"):
        return True
    return summary_like.get("temperature_c_min") is not None or summary_like.get("temperature_c_max") is not None


def _fallback_route_status(
    *,
    metric_key: str,
    route_entry: dict[str, Any],
    requested_mode: str | None,
    actual_mode: str | None,
    temperature_available: bool,
    formal_metric_label: str | None,
    provisional_metric_label: str | None,
    report: Any,
) -> str:
    if route_entry.get("selected_as_formal") or formal_metric_label == metric_key:
        return "formal_passed" if actual_mode == "formal_af" else "provisional"
    if provisional_metric_label == metric_key:
        return "provisional"
    if not temperature_available:
        return "quicklook_only"
    if requested_mode == "quicklook" and actual_mode == "quicklook":
        return "quicklook_only"
    if report is None and route_entry.get("af95_c") is None and route_entry.get("aftan_c") is None:
        return "formal_blocked"
    return "formal_blocked"


def _fallback_route_gate_reason(
    *,
    metric_key: str,
    route_entry: dict[str, Any],
    temperature_available: bool,
    requested_mode: str | None,
    actual_mode: str | None,
    object_gate_reason: str | None,
    report: Any,
    route_status: str,
) -> str | None:
    if route_status in {"formal_passed", "provisional"}:
        return None
    if not temperature_available:
        return "temperature_sync_missing"
    if report is None and route_entry.get("af95_c") is None and route_entry.get("aftan_c") is None:
        return f"{metric_key}_insufficient_points"
    if requested_mode == "formal_af" and actual_mode == "quicklook":
        return object_gate_reason or "legacy_route_status_unavailable"
    return "legacy_route_status_unavailable"


def _normalize_route_results(
    *,
    preset: str | None,
    route_results: Any,
    metric_reports: dict[str, Any] | None,
    requested_mode: str | None,
    actual_mode: str | None,
    temperature_filename: str | None,
    temperature_c_min: float | None = None,
    temperature_c_max: float | None = None,
    primary_metric_label: str | None = None,
    formal_metric_label: str | None = None,
    provisional_metric_label: str | None = None,
    object_gate_reason: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    route_specs = _route_specs_for_preset(preset)
    metric_reports = metric_reports or {}
    normalized_input = {
        "temperature_filename": temperature_filename,
        "temperature_c_min": temperature_c_min,
        "temperature_c_max": temperature_c_max,
    }
    temperature_available = _temperature_available_for_summary(normalized_input)

    raw_entries: list[dict[str, Any]] = []
    if isinstance(route_results, dict):
        for container_key, value in route_results.items():
            entry = dict(value or {})
            entry.setdefault("_container_key", container_key)
            raw_entries.append(entry)
    elif isinstance(route_results, list):
        raw_entries = [dict(item or {}) for item in route_results]

    entries_by_alias: dict[str, dict[str, Any]] = {}
    for raw_entry in raw_entries:
        metric_key = raw_entry.get("metric_key")
        alias = (
            raw_entry.get("alias")
            or raw_entry.get("route_alias")
            or (
                raw_entry.get("_container_key")
                if raw_entry.get("_container_key") in ROUTE_ALIAS_ORDER
                else None
            )
            or _route_alias_from_metric_key(preset, metric_key)
        )
        if alias not in route_specs:
            continue
        merged = dict(entries_by_alias.get(alias, {}))
        merged.update(raw_entry)
        merged["alias"] = alias
        entries_by_alias[alias] = merged

    normalized_results: list[dict[str, Any]] = []
    route_results_by_alias: dict[str, dict[str, Any]] = {}
    for alias in ROUTE_ALIAS_ORDER:
        spec = route_specs[alias]
        entry = dict(entries_by_alias.get(alias, {}))
        metric_key = str(entry.get("metric_key") or spec["metric_key"])
        report = _route_metric_report(metric_reports, spec)
        route_status = entry.get("reportability_status")
        if route_status is None:
            route_status = _fallback_route_status(
                metric_key=metric_key,
                route_entry=entry,
                requested_mode=requested_mode,
                actual_mode=actual_mode,
                temperature_available=temperature_available,
                formal_metric_label=formal_metric_label,
                provisional_metric_label=provisional_metric_label,
                report=report,
            )
        gate_reason = entry.get("gate_reason") or entry.get("formal_gate_reason")
        if gate_reason is None:
            gate_reason = _fallback_route_gate_reason(
                metric_key=metric_key,
                route_entry=entry,
                temperature_available=temperature_available,
                requested_mode=requested_mode,
                actual_mode=actual_mode,
                object_gate_reason=object_gate_reason,
                report=report,
                route_status=route_status,
            )
        warning_codes = _coerce_warning_codes(entry.get("warning_codes"))
        if not warning_codes and gate_reason is not None and route_status not in {"formal_passed", "provisional"}:
            warning_codes = [gate_reason]

        normalized = {
            "alias": alias,
            "route_alias": alias,
            "metric_key": metric_key,
            "display_label": entry.get("display_label") or spec["display_label"],
            "af95_c": _coerce_float(entry.get("af95_c")),
            "aftan_c": _coerce_float(entry.get("aftan_c")),
            "smoothed_af95_c": _coerce_float(entry.get("smoothed_af95_c")),
            "smoothed_aftan_c": _coerce_float(entry.get("smoothed_aftan_c")),
            "smoothing_method": entry.get("smoothing_method"),
            "smoothing_window": entry.get("smoothing_window"),
            "fit_rmse": _coerce_float(entry.get("fit_rmse")),
            "monotonic_violation_fraction": _coerce_float(entry.get("monotonic_violation_fraction")),
            "dynamic_range": _coerce_float(entry.get("dynamic_range")),
            "reportability_status": route_status,
            "reportability_label": _route_status_label(route_status),
            "gate_reason": gate_reason,
            "warning_codes": warning_codes,
            "formal_candidate": bool(entry.get("formal_candidate", spec["formal_candidate"])),
            "accepted_as_formal_candidate": bool(
                entry.get(
                    "accepted_as_formal_candidate",
                    entry.get("selected_as_formal") or (route_status == "formal_passed"),
                )
            ),
            "selected_as_primary": bool(entry.get("selected_as_primary", primary_metric_label == metric_key)),
            "selected_as_formal": bool(entry.get("selected_as_formal", formal_metric_label == metric_key)),
            "formal_role": entry.get("formal_role"),
            "auxiliary_metric_keys": list(entry.get("auxiliary_metric_keys") or []),
            "source": "route_results" if alias in entries_by_alias else "summary_fallback",
        }
        if normalized["af95_c"] is None:
            normalized["af95_c"] = _coerce_float(_metric_report_field(report, "af95_c"))
        if normalized["aftan_c"] is None:
            normalized["aftan_c"] = _coerce_float(_metric_report_field(report, "aftan_c"))
        if normalized["fit_rmse"] is None:
            normalized["fit_rmse"] = _coerce_float(_metric_report_field(report, "fit_rmse"))
        if normalized["monotonic_violation_fraction"] is None:
            normalized["monotonic_violation_fraction"] = _coerce_float(
                _metric_report_field(report, "monotonic_violation_fraction")
            )
        if normalized["dynamic_range"] is None:
            normalized["dynamic_range"] = _coerce_float(_metric_report_field(report, "dynamic_range"))
        if normalized["selected_as_formal"]:
            normalized["formal_role"] = normalized["formal_role"] or "object_formal"
            normalized["accepted_as_formal_candidate"] = True
        elif normalized["selected_as_primary"]:
            normalized["formal_role"] = normalized["formal_role"] or "object_primary"
        elif normalized["formal_candidate"]:
            normalized["formal_role"] = normalized["formal_role"] or "formal_candidate"
        else:
            normalized["formal_role"] = normalized["formal_role"] or "route_result"

        route_results_by_alias[alias] = normalized
        normalized_results.append(normalized)

    return normalized_results, route_results_by_alias


def _prepare_summary_for_display(
    run: sqlite3.Row | dict[str, Any],
    summary: dict[str, Any] | None,
    *,
    backfill_smoothed: bool = True,
) -> dict[str, Any] | None:
    if summary is None:
        return None
    prepared = dict(summary)
    route_results, route_results_by_alias = _normalize_route_results(
        preset=prepared.get("preset") or run["preset"],
        route_results=prepared.get("route_results"),
        metric_reports=prepared.get("metric_reports"),
        requested_mode=prepared.get("requested_mode") or run["requested_mode"],
        actual_mode=prepared.get("actual_mode") or run["actual_mode"],
        temperature_filename=prepared.get("temperature_filename") or run["temperature_filename"],
        temperature_c_min=prepared.get("temperature_c_min"),
        temperature_c_max=prepared.get("temperature_c_max"),
        primary_metric_label=prepared.get("primary_metric_label"),
        formal_metric_label=prepared.get("formal_metric_label"),
        provisional_metric_label=prepared.get("provisional_metric_label"),
        object_gate_reason=prepared.get("formal_gate_reason"),
    )
    prepared["route_results"] = route_results
    prepared["route_results_by_alias"] = route_results_by_alias
    prepared["route_alias_order"] = list(ROUTE_ALIAS_ORDER)
    prepared["route_results_schema_version"] = ROUTE_RESULTS_SCHEMA_VERSION
    prepared["recommended_route_alias"] = None
    if prepared.get("formal_metric_label"):
        prepared["recommended_route_alias"] = _route_alias_from_metric_key(
            prepared.get("preset") or run["preset"],
            prepared.get("formal_metric_label"),
        )
    elif prepared.get("provisional_metric_label"):
        prepared["recommended_route_alias"] = _route_alias_from_metric_key(
            prepared.get("preset") or run["preset"],
            prepared.get("provisional_metric_label"),
        )
    prepared.update(
        canonical_object_result_fields(
            preset=prepared.get("preset") or run["preset"],
            reportability_status=prepared.get("reportability_status"),
            formal_metric_key=prepared.get("formal_metric_label"),
            formal_gate_reason=prepared.get("formal_gate_reason"),
            provisional_metric_key=prepared.get("provisional_metric_label"),
            af95_c=prepared.get("af95_c"),
            aftan_c=prepared.get("aftan_c"),
            provisional_af95_c=prepared.get("provisional_af95_c"),
            provisional_aftan_c=prepared.get("provisional_aftan_c"),
        )
    )
    direction_result = prepared.get("direction_result")
    if isinstance(direction_result, dict):
        prepared["direction_result"] = dict(direction_result)
    if backfill_smoothed:
        prepared = _backfill_smoothed_summary_fields(run, prepared)
    return _augment_reported_result_fields(prepared)


def _first_finite_float(*values: Any) -> float | None:
    for value in values:
        numeric = _coerce_float(value)
        if numeric is not None:
            return numeric
    return None


def _augment_reported_result_fields(prepared: dict[str, Any]) -> dict[str, Any]:
    route_results_by_alias = prepared.get("route_results_by_alias") or {}
    reported_metric_key = (
        prepared.get("object_formal_metric_key")
        or prepared.get("object_provisional_metric_key")
        or prepared.get("object_smoothed_metric_key")
        or prepared.get("object_recommended_metric_key")
    )
    reported_route_alias = (
        prepared.get("object_formal_route_alias")
        or prepared.get("object_provisional_route_alias")
        or prepared.get("object_smoothed_route_alias")
        or prepared.get("object_recommended_route_alias")
        or prepared.get("recommended_route_alias")
    )
    reported_route = route_results_by_alias.get(reported_route_alias or "", {})
    if reported_metric_key is None:
        reported_metric_key = reported_route.get("metric_key")

    prepared["object_reported_metric_key"] = reported_metric_key
    prepared["object_reported_route_alias"] = reported_route_alias
    prepared["object_reported_af95_c"] = _first_finite_float(
        prepared.get("object_formal_af95_c"),
        prepared.get("object_provisional_af95_c"),
        prepared.get("object_smoothed_af95_c"),
        reported_route.get("af95_c"),
        reported_route.get("smoothed_af95_c"),
    )
    prepared["object_reported_aftan_c"] = _first_finite_float(
        prepared.get("object_formal_aftan_c"),
        prepared.get("object_provisional_aftan_c"),
        prepared.get("object_smoothed_aftan_c"),
        reported_route.get("aftan_c"),
        reported_route.get("smoothed_aftan_c"),
    )
    return prepared


def _route_metric_key_for_alias(preset: str | None, alias: str) -> str | None:
    return _route_specs_for_preset(preset).get(alias, {}).get("metric_key")


def _resolve_output_dir(value: Any, *, fallback: Path) -> Path:
    if value in {None, ""}:
        return fallback.resolve()
    path = Path(str(value))
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _safe_read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _infer_benchmark_family(summary: dict[str, Any]) -> str | None:
    preset = str(summary.get("preset") or "")
    if preset.startswith("braided"):
        return "braided"
    if preset in {"wire_like", "demo"}:
        return "wire"
    if any(key in summary for key in {"metric_aliases", "af_comparison_by_alias", "formal_qc"}):
        return "braided"
    if any(key in summary for key in {"route_aliases", "route_benchmark_overview_by_alias", "wire_qc"}):
        return "wire"
    metric_reports = summary.get("metric_reports") or {}
    if any(key in metric_reports for key in {"length_axis", "diameter_max", "area_proj"}):
        return "braided"
    if any(key in metric_reports for key in {"x_route_a", "kappa_fit", "kappa_route_c"}):
        return "wire"
    return None


def _prepare_summary_like_for_display(summary: dict[str, Any], preset_hint: str) -> dict[str, Any]:
    mapped = dict(summary)
    mapped.setdefault("preset", preset_hint)
    mapped.setdefault("actual_mode", mapped.get("mode"))
    if mapped.get("requested_mode") is None:
        if mapped.get("actual_mode") == "formal_af" or mapped.get("formal_metric_label") or mapped.get("provisional_metric_label"):
            mapped["requested_mode"] = "formal_af"
        else:
            mapped["requested_mode"] = "quicklook"
    fake_run = {
        "id": mapped.get("run_id") or mapped.get("benchmark_name") or "benchmark",
        "preset": mapped.get("preset") or preset_hint,
        "requested_mode": mapped.get("requested_mode"),
        "actual_mode": mapped.get("actual_mode"),
        "temperature_filename": mapped.get("temperature_filename"),
    }
    prepared = _prepare_summary_for_display(fake_run, mapped)
    return prepared or mapped


def _benchmark_detail_lookup(family: str) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in sorted(PROJECT_OUTPUTS_ROOT.glob("*/analysis_metrics.json")):
        raw = _safe_read_json(path)
        if not isinstance(raw, dict):
            continue
        if _infer_benchmark_family(raw) != family:
            continue
        output_dir = _resolve_output_dir(raw.get("demo_output"), fallback=path.parent)
        benchmark_name = str(raw.get("benchmark_name") or output_dir.name)
        entries.append(
            {
                "benchmark_name": benchmark_name,
                "output_dir": output_dir,
                "summary": raw,
                "path": path.resolve(),
            }
        )
    by_name = {entry["benchmark_name"]: entry for entry in entries}
    by_output_dir = {str(entry["output_dir"]): entry for entry in entries}
    by_dir_name = {entry["output_dir"].name: entry for entry in entries}
    return {
        "entries": entries,
        "by_name": by_name,
        "by_output_dir": by_output_dir,
        "by_dir_name": by_dir_name,
    }


def _match_benchmark_detail(row: dict[str, Any], lookup: dict[str, Any]) -> dict[str, Any] | None:
    output_dir_value = row.get("output_dir") or row.get("demo_output")
    if output_dir_value not in {None, ""}:
        resolved = _resolve_output_dir(output_dir_value, fallback=PROJECT_OUTPUTS_ROOT)
        detail = lookup["by_output_dir"].get(str(resolved))
        if detail is not None:
            return detail
        detail = lookup["by_dir_name"].get(resolved.name)
        if detail is not None:
            return detail
    benchmark_name = row.get("benchmark_name")
    if benchmark_name is not None:
        detail = lookup["by_name"].get(str(benchmark_name))
        if detail is not None:
            return detail
    return None


def _wire_benchmark_comparison_by_alias(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    overview = summary.get("route_benchmark_overview_by_alias")
    if isinstance(overview, dict) and overview:
        return {alias: dict(entry or {}) for alias, entry in overview.items()}

    comparisons: dict[str, dict[str, Any]] = {}
    has_suite_fields = any(summary.get(f"route_{alias}_metric_key") is not None for alias in ROUTE_ALIAS_ORDER)
    if has_suite_fields:
        for alias in ROUTE_ALIAS_ORDER:
            prefix = f"route_{alias}"
            comparisons[alias] = {
                "alias": alias,
                "metric_key": summary.get(f"{prefix}_metric_key"),
                "display_label": summary.get(f"{prefix}_display_label"),
                "available": summary.get(f"{prefix}_available"),
                "reportability_status": summary.get(f"{prefix}_status"),
                "gate_reason": summary.get(f"{prefix}_gate_reason"),
                "accepted_as_formal_candidate": summary.get(f"{prefix}_accepted_as_formal_candidate"),
                "selected_as_primary": summary.get(f"{prefix}_selected_as_primary"),
                "selected_as_formal": summary.get(f"{prefix}_selected_as_formal"),
                "measured_af95_c": summary.get(f"{prefix}_af95_c"),
                "measured_aftan_c": summary.get(f"{prefix}_aftan_c"),
                "truth_af95_c": summary.get(f"{prefix}_truth_af95_c"),
                "truth_aftan_c": summary.get(f"{prefix}_truth_aftan_c"),
                "af95_error_c": summary.get(f"{prefix}_af95_error_c"),
                "aftan_error_c": summary.get(f"{prefix}_aftan_error_c"),
                "fit_rmse": summary.get(f"{prefix}_fit_rmse"),
                "dynamic_range": summary.get(f"{prefix}_dynamic_range"),
                "monotonic_violation_fraction": summary.get(f"{prefix}_monotonic_violation_fraction"),
            }
        return comparisons

    metric_reports = summary.get("metric_reports") or {}
    truth_metrics = summary.get("truth_metrics") or {}
    truth_comparison = summary.get("metric_truth_comparison") or {}
    for alias, (metric_key, comparison_key, truth_af95_key, truth_aftan_key) in WIRE_LEGACY_ROUTE_METRICS.items():
        report = metric_reports.get(metric_key) or {}
        comparison = truth_comparison.get(comparison_key) or {}
        comparisons[alias] = {
            "alias": alias,
            "metric_key": metric_key,
            "display_label": _route_specs_for_preset("demo")[alias]["display_label"],
            "available": report.get("af95_c") is not None and report.get("aftan_c") is not None,
            "measured_af95_c": report.get("af95_c"),
            "measured_aftan_c": report.get("aftan_c"),
            "truth_af95_c": truth_metrics.get(truth_af95_key),
            "truth_aftan_c": truth_metrics.get(truth_aftan_key),
            "af95_error_c": comparison.get("af95_error_c"),
            "aftan_error_c": comparison.get("aftan_error_c"),
            "fit_rmse": report.get("fit_rmse"),
            "dynamic_range": report.get("dynamic_range"),
            "monotonic_violation_fraction": report.get("monotonic_violation_fraction"),
        }
    return comparisons


def _braided_benchmark_comparison_by_alias(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    comparisons = summary.get("af_comparison_by_alias")
    if isinstance(comparisons, dict) and comparisons:
        return {alias: dict(entry or {}) for alias, entry in comparisons.items()}

    suite_comparisons: dict[str, dict[str, Any]] = {}
    for alias, (metric_key, af95_key, aftan_key) in BRAIDED_SUITE_ROUTE_FIELDS.items():
        if not any(key in summary for key in {af95_key, aftan_key}):
            continue
        suite_comparisons[alias] = {
            "alias": alias,
            "metric_key": metric_key,
            "display_label": _route_specs_for_preset("braided_demo")[alias]["display_label"],
            "af95_error_c": summary.get(af95_key),
            "aftan_error_c": summary.get(aftan_key),
        }
    return suite_comparisons


def _benchmark_qc_highlights(summary: dict[str, Any], family: str) -> list[dict[str, Any]]:
    if family == "wire":
        wire_qc = summary.get("wire_qc") or {}
        return [
            {"label": "质量中位数", "value": summary.get("quality_median", wire_qc.get("quality_median"))},
            {
                "label": "A 路线端点跳变 P95",
                "value": wire_qc.get("route_a_endpoint_jump_p95_px"),
                "suffix": "px",
            },
            {
                "label": "中心线点数中位数",
                "value": wire_qc.get("centerline_points_median"),
            },
            {"label": "二次拟合占比", "value": wire_qc.get("quadratic_fraction")},
        ]
    return [
        {"label": "质量中位数", "value": summary.get("quality_median")},
        {
            "label": "附件泄漏比例",
            "value": summary.get("body_mask_attachment_leak_fraction"),
        },
        {
            "label": "中心线分歧",
            "value": summary.get("centerline_disagreement_median"),
        },
        {
            "label": "端点跳变 P95",
            "value": summary.get("endpoint_jump_p95_px"),
            "suffix": "px",
        },
    ]


def _merge_benchmark_route(
    *,
    alias: str,
    preset: str,
    prepared_route: dict[str, Any] | None,
    comparison_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    spec = _route_specs_for_preset(preset)[alias]
    route = dict(prepared_route or {})
    comparison_entry = comparison_entry or {}
    metric_key = comparison_entry.get("metric_key") or route.get("metric_key") or spec["metric_key"]
    measured_af95 = _coerce_float(comparison_entry.get("measured_af95_c"))
    if measured_af95 is None:
        measured_af95 = _coerce_float(route.get("af95_c"))
    measured_aftan = _coerce_float(comparison_entry.get("measured_aftan_c"))
    if measured_aftan is None:
        measured_aftan = _coerce_float(route.get("aftan_c"))
    truth_af95 = _coerce_float(comparison_entry.get("truth_af95_c"))
    truth_aftan = _coerce_float(comparison_entry.get("truth_aftan_c"))
    af95_error = _coerce_float(comparison_entry.get("af95_error_c"))
    if af95_error is None and measured_af95 is not None and truth_af95 is not None:
        af95_error = measured_af95 - truth_af95
    aftan_error = _coerce_float(comparison_entry.get("aftan_error_c"))
    if aftan_error is None and measured_aftan is not None and truth_aftan is not None:
        aftan_error = measured_aftan - truth_aftan
    reportability_status = route.get("reportability_status") or comparison_entry.get("reportability_status")
    reportability_label = route.get("reportability_label")
    if reportability_label is None and reportability_status is not None:
        reportability_label = _route_status_label(reportability_status)

    return {
        **route,
        "alias": alias,
        "metric_key": metric_key,
        "display_label": comparison_entry.get("display_label") or route.get("display_label") or spec["display_label"],
        "reportability_status": reportability_status,
        "reportability_label": reportability_label or "-",
        "gate_reason": route.get("gate_reason") or comparison_entry.get("gate_reason"),
        "accepted_as_formal_candidate": bool(
            route.get("accepted_as_formal_candidate", comparison_entry.get("accepted_as_formal_candidate"))
        ),
        "selected_as_primary": bool(route.get("selected_as_primary", comparison_entry.get("selected_as_primary"))),
        "selected_as_formal": bool(route.get("selected_as_formal", comparison_entry.get("selected_as_formal"))),
        "formal_candidate": bool(route.get("formal_candidate", spec["formal_candidate"])),
        "measured_af95_c": measured_af95,
        "measured_aftan_c": measured_aftan,
        "truth_af95_c": truth_af95,
        "truth_aftan_c": truth_aftan,
        "af95_error_c": af95_error,
        "aftan_error_c": aftan_error,
        "abs_af95_error_c": None if af95_error is None else abs(af95_error),
        "abs_aftan_error_c": None if aftan_error is None else abs(aftan_error),
        "fit_rmse": _coerce_float(comparison_entry.get("fit_rmse") or route.get("fit_rmse")),
        "dynamic_range": _coerce_float(comparison_entry.get("dynamic_range") or route.get("dynamic_range")),
        "monotonic_violation_fraction": _coerce_float(
            comparison_entry.get("monotonic_violation_fraction") or route.get("monotonic_violation_fraction")
        ),
        "available": bool(
            comparison_entry.get("available")
            if comparison_entry.get("available") is not None
            else measured_af95 is not None or measured_aftan is not None
        ),
    }


def _normalize_benchmark_entry(
    *,
    family: str,
    row: dict[str, Any],
    detail: dict[str, Any] | None,
    source_kind: str,
) -> dict[str, Any]:
    preset_hint = str(BENCHMARK_FAMILY_DISPLAY[family]["preset"])
    detail_summary = detail["summary"] if detail is not None else None
    merged = dict(detail_summary or {})
    for key, value in row.items():
        current = merged.get(key)
        if key not in merged or current is None or current == "" or current == [] or current == {}:
            merged[key] = value
    merged.setdefault("benchmark_name", row.get("benchmark_name") or merged.get("benchmark_name"))
    merged.setdefault("benchmark_description", row.get("description") or merged.get("benchmark_description"))
    merged.setdefault("demo_output", row.get("output_dir") or merged.get("demo_output"))
    prepared = _prepare_summary_like_for_display(merged, preset_hint)
    comparisons = (
        _wire_benchmark_comparison_by_alias(merged) if family == "wire" else _braided_benchmark_comparison_by_alias(merged)
    )

    routes: list[dict[str, Any]] = []
    routes_by_alias = prepared.get("route_results_by_alias") or {}
    for alias in ROUTE_ALIAS_ORDER:
        routes.append(
            _merge_benchmark_route(
                alias=alias,
                preset=preset_hint,
                prepared_route=routes_by_alias.get(alias),
                comparison_entry=comparisons.get(alias),
            )
        )

    output_dir = _resolve_output_dir(
        merged.get("demo_output") or merged.get("output_dir"),
        fallback=detail["output_dir"] if detail is not None else PROJECT_OUTPUTS_ROOT,
    )
    source_label = (
        "汇总文件 + analysis_metrics"
        if source_kind == "suite_summary" and detail_summary is not None
        else "汇总文件"
        if source_kind == "suite_summary"
        else "analysis_metrics"
    )
    return {
        "family": family,
        "benchmark_name": str(merged.get("benchmark_name") or output_dir.name),
        "description": merged.get("benchmark_description") or merged.get("description") or "",
        "output_dir": str(output_dir),
        "output_dir_name": output_dir.name,
        "source_kind": source_kind,
        "source_label": source_label,
        "object_reportability_status": prepared.get("object_reportability_status"),
        "object_formal_metric_key": prepared.get("object_formal_metric_key"),
        "object_formal_route_alias": prepared.get("object_formal_route_alias"),
        "object_formal_gate_reason": prepared.get("object_formal_gate_reason"),
        "object_formal_af95_c": prepared.get("object_formal_af95_c"),
        "object_formal_aftan_c": prepared.get("object_formal_aftan_c"),
        "object_provisional_metric_key": prepared.get("object_provisional_metric_key"),
        "object_provisional_route_alias": prepared.get("object_provisional_route_alias"),
        "object_provisional_af95_c": prepared.get("object_provisional_af95_c"),
        "object_provisional_aftan_c": prepared.get("object_provisional_aftan_c"),
        "object_recommended_metric_key": prepared.get("object_recommended_metric_key"),
        "object_recommended_route_alias": prepared.get("object_recommended_route_alias"),
        "warning_codes": prepared.get("warning_codes") or merged.get("warning_codes") or [],
        "requested_mode": prepared.get("requested_mode"),
        "actual_mode": prepared.get("actual_mode"),
        "routes": routes,
        "routes_by_alias": {entry["alias"]: entry for entry in routes},
        "qc_highlights": _benchmark_qc_highlights(merged, family),
    }


def _build_family_route_rollups(family: str, benchmarks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preset_hint = str(BENCHMARK_FAMILY_DISPLAY[family]["preset"])
    rollups: list[dict[str, Any]] = []
    for alias in ROUTE_ALIAS_ORDER:
        spec = _route_specs_for_preset(preset_hint)[alias]
        entries = [benchmark["routes_by_alias"][alias] for benchmark in benchmarks if alias in benchmark["routes_by_alias"]]
        status_counts = Counter((entry.get("reportability_status") or "unrecorded") for entry in entries)
        gate_counts = Counter(str(entry.get("gate_reason")) for entry in entries if entry.get("gate_reason"))
        rollups.append(
            {
                "alias": alias,
                "display_label": spec["display_label"],
                "metric_key": spec["metric_key"],
                "benchmark_count": len(entries),
                "available_count": sum(1 for entry in entries if entry.get("available")),
                "formal_passed_count": status_counts.get("formal_passed", 0),
                "provisional_count": status_counts.get("provisional", 0),
                "formal_blocked_count": status_counts.get("formal_blocked", 0),
                "quicklook_only_count": status_counts.get("quicklook_only", 0),
                "unrecorded_count": status_counts.get("unrecorded", 0),
                "selected_as_formal_count": sum(1 for entry in entries if entry.get("selected_as_formal")),
                "selected_as_primary_count": sum(1 for entry in entries if entry.get("selected_as_primary")),
                "mean_abs_af95_error_c": _mean_or_none(
                    [entry["abs_af95_error_c"] for entry in entries if entry.get("abs_af95_error_c") is not None]
                ),
                "mean_abs_aftan_error_c": _mean_or_none(
                    [entry["abs_aftan_error_c"] for entry in entries if entry.get("abs_aftan_error_c") is not None]
                ),
                "mean_fit_rmse": _mean_or_none(
                    [entry["fit_rmse"] for entry in entries if entry.get("fit_rmse") is not None]
                ),
                "mean_monotonic_violation_fraction": _mean_or_none(
                    [
                        entry["monotonic_violation_fraction"]
                        for entry in entries
                        if entry.get("monotonic_violation_fraction") is not None
                    ]
                ),
                "dominant_gate_reason": gate_counts.most_common(1)[0][0] if gate_counts else None,
            }
        )
    return rollups


def _build_benchmark_family_view(family: str) -> dict[str, Any]:
    lookup = _benchmark_detail_lookup(family)
    suite_path = Path(BENCHMARK_FAMILY_DISPLAY[family]["suite_path"])
    suite_rows = _safe_read_json(suite_path) if suite_path.exists() else None
    benchmarks: list[dict[str, Any]] = []
    used_outputs: set[str] = set()

    if isinstance(suite_rows, list) and suite_rows:
        for raw_row in suite_rows:
            if not isinstance(raw_row, dict):
                continue
            detail = _match_benchmark_detail(raw_row, lookup)
            if detail is not None:
                used_outputs.add(str(detail["output_dir"]))
            benchmarks.append(
                _normalize_benchmark_entry(
                    family=family,
                    row=raw_row,
                    detail=detail,
                    source_kind="suite_summary",
                )
            )

    for detail in lookup["entries"]:
        if str(detail["output_dir"]) in used_outputs:
            continue
        benchmarks.append(
            _normalize_benchmark_entry(
                family=family,
                row=detail["summary"],
                detail=detail,
                source_kind="analysis_metrics",
            )
        )

    if not isinstance(suite_rows, list):
        benchmarks.sort(key=lambda item: item["benchmark_name"])

    route_rollups = _build_family_route_rollups(family, benchmarks)
    return {
        "key": family,
        "label": BENCHMARK_FAMILY_DISPLAY[family]["label"],
        "suite_available": bool(isinstance(suite_rows, list) and suite_rows),
        "suite_path": str(suite_path),
        "source_label": "优先读取汇总文件" if isinstance(suite_rows, list) and suite_rows else "回退到 analysis_metrics",
        "benchmark_count": len(benchmarks),
        "benchmarks": benchmarks,
        "route_rollups": route_rollups,
        "route_rollups_by_alias": {entry["alias"]: entry for entry in route_rollups},
    }


def _build_benchmark_page_data() -> dict[str, Any]:
    families = {family: _build_benchmark_family_view(family) for family in BENCHMARK_FAMILY_DISPLAY}
    comparison_rows = [
        {
            "alias": alias,
            "wire": families["wire"]["route_rollups_by_alias"].get(alias),
            "braided": families["braided"]["route_rollups_by_alias"].get(alias),
        }
        for alias in ROUTE_ALIAS_ORDER
    ]
    return {
        "families": families,
        "comparison_rows": comparison_rows,
    }


def _build_run_card(run: sqlite3.Row) -> dict[str, Any]:
    payload = dict(run)
    summary = _prepare_summary_for_display(payload, _load_summary(payload["id"]), backfill_smoothed=False)
    payload["summary"] = summary
    if summary is not None:
        payload["reportability_status"] = summary.get("reportability_status")
        payload["annotated_video_filename"] = summary.get("annotated_video_filename")
        payload["analyzed_frame_count"] = summary.get("analyzed_frame_count")
        payload["original_frame_count"] = summary.get("original_frame_count")
    payload["requested_mode"] = payload.get("requested_mode") or _requested_mode_for_run(payload.get("temperature_filename"))
    payload["status_label"] = _run_status_label(payload.get("status"))
    payload["can_delete"] = payload.get("status") not in {"queued", "running"}
    return payload


def _run_detail_refresh_needed(run: sqlite3.Row | dict[str, Any], summary: dict[str, Any] | None) -> bool:
    status = run.get("status") if isinstance(run, dict) else run["status"]
    return status in {"queued", "running"} or bool(
        summary and summary.get("asset_generation_status") in {"pending", "running"}
    )


def _run_status_payload(run: sqlite3.Row | dict[str, Any], summary: dict[str, Any] | None) -> dict[str, Any]:
    run_payload = dict(run)
    summary_payload = summary if isinstance(summary, dict) else {}
    refresh = _run_detail_refresh_needed(run_payload, summary_payload)
    return {
        "id": run_payload["id"],
        "status": run_payload["status"],
        "status_label": _run_status_label(run_payload["status"]),
        "asset_generation_status": summary_payload.get("asset_generation_status"),
        "asset_generation_error": summary_payload.get("asset_generation_error"),
        "error_text": run_payload.get("error_text"),
        "refresh": refresh,
        "active": refresh,
    }


templates.env.globals.update(
    preset_label=_preset_label,
    preset_description=_preset_description,
    mode_label=_mode_label,
    mode_description=_mode_description,
    run_status_label=_run_status_label,
    route_formal_role_label=_route_formal_role_label,
    requested_flow_label=_requested_flow_label,
    run_result_hint=_run_result_hint,
)


@app.get("/")
def home(request: Request) -> Any:
    runs = [_build_run_card(run) for run in _list_runs(limit=12)]
    return templates.TemplateResponse(
        "index.html",
        {
            "runs": runs,
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
    runs = [_build_run_card(run) for run in _list_runs(limit=200)]
    status_counts = Counter(run["status"] for run in runs)
    return templates.TemplateResponse(
        "history.html",
        {
            "runs": runs,
            "history_stats": {
                "total": len(runs),
                "completed": status_counts.get("completed", 0),
                "running": status_counts.get("running", 0) + status_counts.get("queued", 0),
                "failed": status_counts.get("failed", 0),
            },
            "deleted": request.query_params.get("deleted") == "1",
            "preset_label": _preset_label,
            "mode_label": _mode_label,
            "run_status_label": _run_status_label,
            "run_result_hint": _run_result_hint,
            "request": request,
        },
    )


@app.get("/benchmarks")
def benchmark_summary(request: Request) -> Any:
    benchmark_page = _build_benchmark_page_data()
    return templates.TemplateResponse(
        "benchmark_summary.html",
        {
            "request": request,
            "benchmark_page": benchmark_page,
            "comparison_rows": benchmark_page["comparison_rows"],
            "families": benchmark_page["families"],
        },
    )


@app.post("/runs")
async def create_run(
    request: Request,
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    temperature_file: UploadFile | None = File(None),
    preset: str = Form("wire_like"),
    frame_stride: str = Form("1"),
    run_name: str = Form(""),
    direction_angle_deg: str = Form(""),
) -> RedirectResponse:
    _ensure_storage()
    if preset not in {"wire_like", "braided_like"}:
        raise HTTPException(status_code=400, detail="unsupported preset")
    if not video_file.filename:
        raise HTTPException(status_code=400, detail="video file is required")
    try:
        parsed_frame_stride = int(frame_stride)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="frame_stride must be an integer") from exc
    if parsed_frame_stride < 1:
        raise HTTPException(status_code=400, detail="frame_stride must be >= 1")
    parsed_direction_angle: float | None = None
    if direction_angle_deg.strip():
        try:
            parsed_direction_angle = float(direction_angle_deg)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="direction_angle_deg must be numeric") from exc
        if not np.isfinite(parsed_direction_angle):
            raise HTTPException(status_code=400, detail="direction_angle_deg must be finite")
    if preset == "braided_like" and parsed_direction_angle is None:
        raise HTTPException(status_code=400, detail="direction_angle_deg is required for braided uploads")

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
    requested_mode = _requested_mode_for_run(temperature_path.name if temperature_path else None)

    _insert_run(
        {
            "id": run_id,
            "created_at": _utc_now(),
            "status": "queued",
            "run_name": run_name.strip() or None,
            "preset": preset,
            "requested_mode": requested_mode,
            "frame_stride": parsed_frame_stride,
            "direction_angle_deg": parsed_direction_angle,
            "direction_metric_enabled": bool(preset.startswith("braided") and parsed_direction_angle is not None),
            "actual_mode": None,
            "formal_metric_label": None,
            "formal_gate_reason": None,
            "af95_c": None,
            "aftan_c": None,
            "original_frame_count": None,
            "analyzed_frame_count": None,
            "annotated_video_filename": None,
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
            "frame_stride": 1,
            "actual_mode": None,
            "formal_metric_label": None,
            "formal_gate_reason": None,
            "af95_c": None,
            "aftan_c": None,
            "original_frame_count": None,
            "analyzed_frame_count": None,
            "annotated_video_filename": None,
            "video_filename": sample_input["video_filename"],
            "temperature_filename": sample_input["temperature_filename"],
            "run_dir": str(run_dir),
            "error_text": None,
        }
    )
    background_tasks.add_task(_execute_run, run_id)
    return RedirectResponse(url=str(request.url_for("run_detail", run_id=run_id)), status_code=303)


@app.post("/preview-video")
async def create_video_preview(
    request: Request,
    video_file: UploadFile = File(...),
) -> dict[str, Any]:
    _ensure_storage()
    if not video_file.filename:
        raise HTTPException(status_code=400, detail="video file is required")

    preview_id = _new_run_id()
    preview_dir = PREVIEWS_ROOT / preview_id
    preview_dir.mkdir(parents=True, exist_ok=True)
    source_path = preview_dir / _safe_filename(video_file.filename)
    await _save_upload(video_file, source_path)

    preview_path = preview_dir / "preview.mp4"
    try:
        preview_path = _transcode_preview_video(source_path, preview_path)
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(preview_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=f"preview generation failed: {exc}") from exc

    poster_url = None
    poster_path = _ensure_video_poster(preview_path)
    if poster_path is not None:
        poster_rel = poster_path.relative_to(DATA_ROOT).as_posix()
        poster_url = str(request.url_for("files", path=poster_rel))
    preview_rel = preview_path.relative_to(DATA_ROOT).as_posix()
    return {
        "preview_id": preview_id,
        "preview_url": str(request.url_for("files", path=preview_rel)),
        "poster_url": poster_url,
    }


@app.post("/runs/{run_id}/delete")
def delete_run(request: Request, run_id: str) -> RedirectResponse:
    run = _get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    if run["status"] in {"queued", "running"} or _asset_generation_active(_load_summary(run_id)):
        raise HTTPException(status_code=409, detail="run is still active")
    _delete_run_assets(run_id, Path(run["run_dir"]))
    _delete_run_record(run_id)
    return RedirectResponse(url=f"{request.url_for('history')}?deleted=1", status_code=303)


@app.get("/runs/{run_id}/status")
def run_status(run_id: str) -> dict[str, Any]:
    run_row = _get_run(run_id)
    if run_row is None:
        raise HTTPException(status_code=404, detail="run not found")
    return _run_status_payload(run_row, _load_summary(run_id))


@app.get("/runs/{run_id}")
def run_detail(request: Request, run_id: str) -> Any:
    run_row = _get_run(run_id)
    if run_row is None:
        raise HTTPException(status_code=404, detail="run not found")
    run = dict(run_row)

    summary = _prepare_summary_for_display(run, _load_summary(run_id), backfill_smoothed=False)
    temperature_available = bool(
        run.get("temperature_filename")
        or (summary and (summary.get("temperature_c_min") is not None or summary.get("temperature_c_max") is not None))
    )
    image_files = []
    curve_files = []
    other_image_files = []
    video_files = []
    download_files = []
    outputs_dir = RUNS_ROOT / run_id / "outputs"
    if outputs_dir.exists():
        for path in sorted(outputs_dir.iterdir()):
            rel = path.relative_to(DATA_ROOT).as_posix()
            if path.suffix.lower() in {".mp4", ".webm"}:
                poster_url = None
                poster_path = path.with_name(f"{path.stem}_poster.jpg")
                if poster_path.exists():
                    poster_rel = poster_path.relative_to(DATA_ROOT).as_posix()
                    poster_url = str(request.url_for("files", path=poster_rel))
                payload = {
                    "name": path.name,
                    "label": _worker_output_label(path.name, run["preset"]),
                    "url": str(request.url_for("files", path=rel)),
                    "poster_url": poster_url,
                }
                video_files.append(payload)
                download_files.append(payload)
                continue
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                payload = {
                    "name": path.name,
                    "label": _worker_output_label(path.name, run["preset"]),
                    "url": str(request.url_for("files", path=rel)),
                }
                image_files.append(payload)
                if "route_" in path.name or "recovery" in path.name or "temperature" in path.name:
                    curve_files.append(payload)
                else:
                    other_image_files.append(payload)
                download_files.append(payload)
                continue
            if path.suffix.lower() in {".csv", ".json"}:
                download_files.append(
                    {
                        "name": path.name,
                        "label": _worker_output_label(path.name, run["preset"]),
                        "url": str(request.url_for("files", path=rel)),
                    }
                )
    annotated_video = None
    process_video = None
    if summary and summary.get("annotated_video_filename"):
        annotated_video = next(
            (item for item in video_files if item["name"] == summary["annotated_video_filename"]),
            None,
        )
    if summary and summary.get("process_video_filename"):
        process_video = next(
            (item for item in video_files if item["name"] == summary["process_video_filename"]),
            None,
        )
    primary_video = process_video or annotated_video or (video_files[0] if video_files else None)
    primary_curve = None
    if curve_files:
        primary_curve = sorted(curve_files, key=lambda item: _curve_priority(item["name"], temperature_available))[0]
    route_curve_names = (
        [
            "route_a_recovery_vs_temperature.png",
            "route_b_recovery_vs_temperature.png",
            "route_c_recovery_vs_temperature.png",
        ]
        if temperature_available
        else [
            "route_a_metric_over_time.png",
            "route_b_metric_over_time.png",
            "route_c_metric_over_time.png",
        ]
    )
    route_curve_files = []
    for filename in route_curve_names:
        matched = next((item for item in curve_files if item["name"] == filename), None)
        if matched is not None:
            route_curve_files.append(matched)
    direction_metric_curve = next((item for item in curve_files if item["name"] == "direction_metric_over_time.png"), None)
    direction_recovery_curve = next((item for item in curve_files if item["name"] == "direction_recovery_vs_temperature.png"), None)

    return templates.TemplateResponse(
        "run_detail.html",
        {
            "request": request,
            "run": run,
            "summary": summary,
            "temperature_available": temperature_available,
            "image_files": image_files,
            "curve_files": curve_files,
            "other_image_files": other_image_files,
            "video_files": video_files,
            "annotated_video": annotated_video,
            "process_video": process_video,
            "primary_video": primary_video,
            "primary_curve": primary_curve,
            "route_curve_files": route_curve_files,
            "direction_result": summary.get("direction_result") if summary else None,
            "direction_metric_curve": direction_metric_curve,
            "direction_recovery_curve": direction_recovery_curve,
            "download_files": download_files,
            "refresh": _run_detail_refresh_needed(run, summary),
            "preset_label": _preset_label,
            "preset_description": _preset_description,
            "mode_label": _mode_label,
            "mode_description": _mode_description,
            "run_status_label": _run_status_label,
            "route_formal_role_label": _route_formal_role_label,
            "requested_flow_label": _requested_flow_label,
            "run_result_hint": _run_result_hint,
            "worker_object_label": _worker_object_label,
            "worker_route_name": _worker_route_name,
            "worker_route_title": _worker_route_title,
            "worker_route_description": _worker_route_description,
            "worker_route_status_label": _worker_route_status_label,
            "worker_route_note": _worker_route_note,
            "worker_result_summary": _worker_result_summary,
            "worker_primary_route_label": _worker_primary_route_label,
            "display_time": _display_time,
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


for _prefix in PATH_PREFIX_ALIASES:
    app.add_api_route(f"{_prefix}/", home, methods=["GET"], include_in_schema=False, name=f"home{_prefix}")
    app.add_api_route(f"{_prefix}/history", history, methods=["GET"], include_in_schema=False, name=f"history{_prefix}")
    app.add_api_route(
        f"{_prefix}/benchmarks",
        benchmark_summary,
        methods=["GET"],
        include_in_schema=False,
        name=f"benchmark_summary{_prefix}",
    )
    app.add_api_route(f"{_prefix}/runs", create_run, methods=["POST"], include_in_schema=False, name=f"create_run{_prefix}")
    app.add_api_route(
        f"{_prefix}/sample-runs",
        create_sample_run,
        methods=["POST"],
        include_in_schema=False,
        name=f"create_sample_run{_prefix}",
    )
    app.add_api_route(
        f"{_prefix}/preview-video",
        create_video_preview,
        methods=["POST"],
        include_in_schema=False,
        name=f"create_video_preview{_prefix}",
    )
    app.add_api_route(
        f"{_prefix}/runs/{{run_id}}/status",
        run_status,
        methods=["GET"],
        include_in_schema=False,
        name=f"run_status{_prefix}",
    )
    app.add_api_route(
        f"{_prefix}/runs/{{run_id}}",
        run_detail,
        methods=["GET"],
        include_in_schema=False,
        name=f"run_detail{_prefix}",
    )
    app.add_api_route(
        f"{_prefix}/runs/{{run_id}}/delete",
        delete_run,
        methods=["POST"],
        include_in_schema=False,
        name=f"delete_run{_prefix}",
    )
    app.add_api_route(f"{_prefix}/health", health, methods=["GET"], include_in_schema=False, name=f"health{_prefix}")


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
        if run["temperature_filename"]:
            temperature_path = inputs_dir / run["temperature_filename"]
        frame_stride = int(run["frame_stride"]) if "frame_stride" in run.keys() and run["frame_stride"] is not None else 1

        if run["preset"] in {"wire_like", "demo"}:
            extraction_cfg = _build_wire_extraction_config(config, run["preset"])
            route_c = RouteCConfig(**config["analysis"].get("wire_like_extraction", {}).get("route_c", {}))
            result = analyze_video(
                video_path,
                extraction=extraction_cfg,
                temperature_csv=temperature_path,
                route_c=route_c,
                frame_stride=frame_stride,
            )
        elif run["preset"] in {"braided_like", "braided_demo"}:
            extraction_cfg = _build_braided_extraction_config(config)
            result = analyze_braided_video_quicklook(
                video_path,
                extraction=extraction_cfg,
                temperature_csv=temperature_path,
                acceptance_profile="synthetic" if run["preset"] == "braided_demo" else "real_video",
                frame_stride=frame_stride,
                direction_angle_deg=(
                    float(run["direction_angle_deg"])
                    if "direction_angle_deg" in run.keys() and run["direction_angle_deg"] is not None
                    else None
                ),
            )
        else:
            raise RuntimeError(f"unsupported preset: {run['preset']}")

        result.series.to_csv(outputs_dir / "analysis.csv", index=False)
        annotated_video_path = outputs_dir / ANNOTATED_VIDEO_FILENAME
        _remove_video_with_poster(annotated_video_path)
        summary = _build_summary(run, result)
        summary["annotated_video_filename"] = None
        process_video_path = outputs_dir / PROCESS_VIDEO_FILENAME
        _remove_video_with_poster(process_video_path)
        summary["process_video_filename"] = None
        summary["process_video_error"] = None
        summary["asset_generation_status"] = "pending"
        summary["asset_generation_error"] = None
        _write_summary(outputs_dir / "summary.json", summary)
        route_results_dataframe(summary.get("route_results")).to_csv(outputs_dir / "route_results.csv", index=False)

        _update_run(
            run_id,
            {
                "status": "completed",
                "actual_mode": result.mode,
                "formal_metric_label": _public_formal_metric_label(result),
                "formal_gate_reason": result.formal_gate_reason,
                "af95_c": result.af95_c,
                "aftan_c": result.aftan_c,
                "original_frame_count": result.original_frame_count,
                "analyzed_frame_count": result.analyzed_frame_count,
                "annotated_video_filename": None,
                "error_text": None,
            },
        )
        _schedule_postprocess(
            run_id,
            run=run,
            prepared=summary,
            result=result,
            extraction_cfg=extraction_cfg,
            video_path=video_path,
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
        "reportability_status": result.reportability_status,
        "warning_codes": result.warning_codes or [],
        "acceptance_profile": result.acceptance_profile,
        "formal_metric_label": public_formal_metric_label,
        "formal_gate_reason": result.formal_gate_reason,
        "af95_c": result.af95_c,
        "aftan_c": result.aftan_c,
        "provisional_metric_label": result.provisional_metric_label,
        "provisional_af95_c": result.provisional_af95_c,
        "provisional_aftan_c": result.provisional_aftan_c,
        "frame_stride": result.frame_stride,
        "input_fps": result.input_fps,
        "frames": result.analyzed_frame_count if result.analyzed_frame_count is not None else len(series),
        "original_frame_count": result.original_frame_count,
        "analyzed_frame_count": result.analyzed_frame_count if result.analyzed_frame_count is not None else len(series),
        "quality_median": float(series["quality"].median()) if "quality" in series else None,
        "video_filename": run["video_filename"],
        "temperature_filename": run["temperature_filename"],
        "direction_angle_deg": run["direction_angle_deg"] if "direction_angle_deg" in run.keys() else None,
        "direction_metric_enabled": bool(run["direction_metric_enabled"]) if "direction_metric_enabled" in run.keys() else False,
        "primary_metric_label": result.primary_metric_label,
        "annotated_video_filename": run["annotated_video_filename"] if "annotated_video_filename" in run.keys() else None,
        "process_video_filename": None,
    }
    if result.formal_candidate_gates is not None:
        summary["formal_candidate_gates"] = result.formal_candidate_gates
    if str(run["preset"]).startswith("braided"):
        summary["metric_aliases"] = BRAIDED_METRIC_ALIAS_TO_KEY
        summary["metric_display_labels"] = {
            key: f"{alias}:{key}" for alias, key in BRAIDED_METRIC_ALIAS_TO_KEY.items()
        }
        summary["formal_candidate_metrics"] = [
            {
                "label": metric_label,
                "alias": BRAIDED_METRIC_KEY_TO_ALIAS.get(metric_label),
            }
            for metric_label in BRAIDED_FORMAL_CANDIDATES
        ]
        summary["formal_qc_scope"] = "current braided formal-gate QC snapshot; not an overall A/B/C verdict"
        if public_formal_metric_label is not None:
            summary["formal_metric_alias"] = BRAIDED_METRIC_KEY_TO_ALIAS.get(public_formal_metric_label)
        if result.primary_metric_label is not None:
            summary["primary_metric_alias"] = BRAIDED_METRIC_KEY_TO_ALIAS.get(result.primary_metric_label)
        if result.provisional_metric_label is not None:
            summary["provisional_metric_alias"] = BRAIDED_METRIC_KEY_TO_ALIAS.get(result.provisional_metric_label)
        summary["formal_qc"] = compute_braided_acceptance(
            series,
            acceptance_profile=result.acceptance_profile or "real_video",
        )
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
    if "endpoint_frame_jump_px" in series.columns:
        summary["endpoint_frame_jump_p95_px"] = float(series["endpoint_frame_jump_px"].quantile(0.95))
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
    if result.direction_result is not None:
        summary["direction_result"] = dict(result.direction_result)
    route_results, route_results_by_alias = _normalize_route_results(
        preset=run["preset"],
        route_results=result.route_results,
        metric_reports=result.metric_reports,
        requested_mode=run["requested_mode"],
        actual_mode=result.mode,
        temperature_filename=run["temperature_filename"],
        temperature_c_min=summary.get("temperature_c_min"),
        temperature_c_max=summary.get("temperature_c_max"),
        primary_metric_label=result.primary_metric_label,
        formal_metric_label=public_formal_metric_label,
        provisional_metric_label=result.provisional_metric_label,
        object_gate_reason=result.formal_gate_reason,
    )
    smoothed_route_results = _build_smoothed_route_results(run["preset"], series)
    for route in route_results:
        route.update(smoothed_route_results.get(route["alias"], {}))
    summary["route_results"] = route_results
    summary["route_results_by_alias"] = canonical_route_results_by_alias(route_results)
    summary["route_alias_order"] = list(ROUTE_ALIAS_ORDER)
    summary["route_order"] = list(ROUTE_ALIAS_ORDER)
    summary["route_results_schema_version"] = ROUTE_RESULTS_SCHEMA_VERSION
    if smoothed_route_results:
        summary["route_smoothing_method"] = "centered_rolling_median_temperature"
        summary["route_smoothing_window"] = 7
        preferred_metric_key = public_formal_metric_label or result.provisional_metric_label
        preferred_alias = _route_alias_from_metric_key(run["preset"], preferred_metric_key)
        preferred_smoothed = smoothed_route_results.get(preferred_alias or "")
        if preferred_smoothed is not None:
            summary["object_smoothed_metric_key"] = preferred_metric_key
            summary["object_smoothed_route_alias"] = preferred_alias
            summary["object_smoothed_af95_c"] = preferred_smoothed.get("smoothed_af95_c")
            summary["object_smoothed_aftan_c"] = preferred_smoothed.get("smoothed_aftan_c")
    summary.update(
        canonical_object_result_fields(
            preset=run["preset"],
            reportability_status=result.reportability_status,
            formal_metric_key=public_formal_metric_label,
            formal_gate_reason=result.formal_gate_reason,
            provisional_metric_key=result.provisional_metric_label,
            af95_c=result.af95_c,
            aftan_c=result.aftan_c,
            provisional_af95_c=result.provisional_af95_c,
            provisional_aftan_c=result.provisional_aftan_c,
        )
    )
    return summary


def _write_plots(out_dir: Path, result: AnalysisResult) -> None:
    import matplotlib

    # Runs are executed in FastAPI background threads, so force a non-GUI backend.
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    series = result.series
    if series.empty:
        return

    preferred_font_names: list[str] = []
    for font_path in [
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        candidate = Path(font_path)
        if not candidate.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(candidate))
            font_name = font_manager.FontProperties(fname=str(candidate)).get_name()
        except Exception:  # noqa: BLE001
            continue
        if font_name and font_name not in preferred_font_names:
            preferred_font_names.append(font_name)

    plt.rcParams["font.family"] = preferred_font_names[:1] or ["sans-serif"]
    plt.rcParams["font.sans-serif"] = preferred_font_names + [
        "PingFang SC",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    route_entries = canonical_route_results_by_alias(result.route_results)
    route_titles = _plot_route_titles_for_series(series)

    def _draw_route_temperature_markers(alias: str) -> None:
        entry = route_entries.get(alias) or {}
        af95_c = entry.get("af95_c")
        aftan_c = entry.get("aftan_c")
        if af95_c is not None:
            plt.axvline(af95_c, color="tab:green", linestyle="--", label=f"95%恢复温度 {af95_c:.2f}℃")
        if aftan_c is not None:
            plt.axvline(aftan_c, color="tab:red", linestyle="--", label=f"切线法温度 {aftan_c:.2f}℃")

    if {"time_sec", "x_route_a_px"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["time_sec"], series["x_route_a_px"], label=route_titles["A"], linewidth=2.0, color="#7c3aed")
        plt.xlabel("时间（秒）")
        plt.ylabel("距离（像素）")
        plt.title(f"{route_titles['A']}变化曲线")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "route_a_metric_over_time.png", dpi=160)
        plt.close(fig)

    if {"time_sec", "kappa_fit_px_inv"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["time_sec"], series["kappa_fit_px_inv"], label=route_titles["B"], linewidth=2.0, color="#ea580c")
        if "x_fit_px" in series.columns:
            plt.plot(series["time_sec"], series["x_fit_px"], label="辅助量", linewidth=1.4, alpha=0.35, color="#f59e0b")
        plt.xlabel("时间（秒）")
        plt.ylabel("弯曲程度")
        plt.title(f"{route_titles['B']}变化曲线")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "route_b_metric_over_time.png", dpi=160)
        plt.close(fig)

    if {"time_sec", "kappa_route_c_px_inv"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["time_sec"], series["kappa_route_c_px_inv"], label=route_titles["C"], linewidth=2.0, color="#16a34a")
        if "x_route_c_px" in series.columns:
            plt.plot(series["time_sec"], series["x_route_c_px"], label="辅助量", linewidth=1.4, alpha=0.35, color="#22c55e")
        plt.xlabel("时间（秒）")
        plt.ylabel("弯曲程度")
        plt.title(f"{route_titles['C']}变化曲线")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "route_c_metric_over_time.png", dpi=160)
        plt.close(fig)

    if {"time_sec", "length_axis_px"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["time_sec"], series["length_axis_px"], label=route_titles["A"], linewidth=2.0, color="#2563eb")
        plt.xlabel("时间（秒）")
        plt.ylabel("长度（像素）")
        plt.title(f"{route_titles['A']}变化曲线")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "route_a_metric_over_time.png", dpi=160)
        plt.close(fig)

    if {"time_sec", "diameter_max_px"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["time_sec"], series["diameter_max_px"], label=route_titles["B"], linewidth=2.0, color="#d946ef")
        plt.xlabel("时间（秒）")
        plt.ylabel("宽度（像素）")
        plt.title(f"{route_titles['B']}变化曲线")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "route_b_metric_over_time.png", dpi=160)
        plt.close(fig)

    if {"time_sec", "area_proj_px2"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["time_sec"], series["area_proj_px2"], label=route_titles["C"], linewidth=2.0, color="#16a34a")
        plt.xlabel("时间（秒）")
        plt.ylabel("面积（像素²）")
        plt.title(f"{route_titles['C']}变化曲线")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "route_c_metric_over_time.png", dpi=160)
        plt.close(fig)

    if {"time_sec", "direction_span_px"}.issubset(series.columns) and series["direction_span_px"].notna().any():
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["time_sec"], series["direction_span_px"], label="方向法", linewidth=2.0, color="#c2410c")
        angle_deg = _coerce_float(series["direction_angle_deg"].dropna().iloc[0]) if "direction_angle_deg" in series.columns and series["direction_angle_deg"].notna().any() else None
        title = "方向法变化曲线" if angle_deg is None else f"方向法变化曲线 ({angle_deg:.1f}°)"
        plt.xlabel("时间（秒）")
        plt.ylabel("投影跨度（像素）")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "direction_metric_over_time.png", dpi=160)
        plt.close(fig)

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

    if {"x_route_a_recovery", "kappa_fit_recovery", "kappa_route_c_recovery"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["temperature_c"], series["x_route_a_recovery"], label=route_titles["A"], linewidth=1.8)
        plt.plot(series["temperature_c"], series["kappa_fit_recovery"], label=route_titles["B"], linewidth=1.8)
        plt.plot(series["temperature_c"], series["kappa_route_c_recovery"], label=route_titles["C"], linewidth=1.8)
        if result.mode == "formal_af" and result.af95_c is not None:
            plt.axvline(result.af95_c, color="tab:green", linestyle="--", label=f"95%恢复温度 {result.af95_c:.2f}℃")
        elif result.provisional_af95_c is not None:
            plt.axvline(
                result.provisional_af95_c,
                color="tab:green",
                linestyle=":",
                label=f"参考95%恢复温度 {result.provisional_af95_c:.2f}℃",
            )
        if result.mode == "formal_af" and result.aftan_c is not None:
            plt.axvline(result.aftan_c, color="tab:red", linestyle="--", label=f"切线法温度 {result.aftan_c:.2f}℃")
        elif result.provisional_aftan_c is not None:
            plt.axvline(
                result.provisional_aftan_c,
                color="tab:red",
                linestyle=":",
                label=f"参考切线法温度 {result.provisional_aftan_c:.2f}℃",
            )
        plt.xlabel("温度（℃）")
        plt.ylabel("恢复比例")
        plt.title("三种测量方式温度曲线")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "route_recovery_vs_temperature.png", dpi=160)
        fig.savefig(out_dir / "recovery_vs_temperature.png", dpi=160)
        plt.close(fig)

    if {"temperature_c", "x_route_a_recovery"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["temperature_c"], series["x_route_a_recovery"], label=route_titles["A"], linewidth=2.0, color="#7c3aed")
        _draw_route_temperature_markers("A")
        plt.xlabel("温度（℃）")
        plt.ylabel("恢复比例")
        plt.title(f"{route_titles['A']}温度曲线")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "route_a_recovery_vs_temperature.png", dpi=160)
        plt.close(fig)

    if {"temperature_c", "kappa_fit_recovery"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["temperature_c"], series["kappa_fit_recovery"], label=route_titles["B"], linewidth=2.0, color="#ea580c")
        _draw_route_temperature_markers("B")
        plt.xlabel("温度（℃）")
        plt.ylabel("恢复比例")
        plt.title(f"{route_titles['B']}温度曲线")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "route_b_recovery_vs_temperature.png", dpi=160)
        plt.close(fig)

    if {"temperature_c", "kappa_route_c_recovery"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["temperature_c"], series["kappa_route_c_recovery"], label=route_titles["C"], linewidth=2.0, color="#16a34a")
        _draw_route_temperature_markers("C")
        plt.xlabel("温度（℃）")
        plt.ylabel("恢复比例")
        plt.title(f"{route_titles['C']}温度曲线")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "route_c_recovery_vs_temperature.png", dpi=160)
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

    if {"temperature_c", "length_axis_recovery", "diameter_max_recovery", "area_proj_recovery"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["temperature_c"], series["length_axis_recovery"], label=route_titles["A"], linewidth=1.8)
        plt.plot(series["temperature_c"], series["diameter_max_recovery"], label=route_titles["B"], linewidth=1.8)
        plt.plot(series["temperature_c"], series["area_proj_recovery"], label=route_titles["C"], linewidth=1.8)
        if result.mode == "formal_af" and result.af95_c is not None:
            plt.axvline(result.af95_c, color="tab:green", linestyle="--", label=f"95%恢复温度 {result.af95_c:.2f}℃")
        elif result.provisional_af95_c is not None:
            plt.axvline(
                result.provisional_af95_c,
                color="tab:green",
                linestyle=":",
                label=f"参考95%恢复温度 {result.provisional_af95_c:.2f}℃",
            )
        if result.mode == "formal_af" and result.aftan_c is not None:
            plt.axvline(result.aftan_c, color="tab:red", linestyle="--", label=f"切线法温度 {result.aftan_c:.2f}℃")
        elif result.provisional_aftan_c is not None:
            plt.axvline(
                result.provisional_aftan_c,
                color="tab:red",
                linestyle=":",
                label=f"参考切线法温度 {result.provisional_aftan_c:.2f}℃",
            )
        plt.xlabel("温度（℃）")
        plt.ylabel("恢复比例")
        plt.title("三种测量方式温度曲线")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "route_recovery_vs_temperature.png", dpi=160)
        fig.savefig(out_dir / "braided_recovery_vs_temperature.png", dpi=160)
        plt.close(fig)

    if {"temperature_c", "length_axis_recovery"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["temperature_c"], series["length_axis_recovery"], label=route_titles["A"], linewidth=2.0, color="#2563eb")
        _draw_route_temperature_markers("A")
        plt.xlabel("温度（℃）")
        plt.ylabel("恢复比例")
        plt.title(f"{route_titles['A']}温度曲线")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "route_a_recovery_vs_temperature.png", dpi=160)
        plt.close(fig)

    if {"temperature_c", "diameter_max_recovery"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["temperature_c"], series["diameter_max_recovery"], label=route_titles["B"], linewidth=2.0, color="#d946ef")
        _draw_route_temperature_markers("B")
        plt.xlabel("温度（℃）")
        plt.ylabel("恢复比例")
        plt.title(f"{route_titles['B']}温度曲线")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "route_b_recovery_vs_temperature.png", dpi=160)
        plt.close(fig)

    if {"temperature_c", "area_proj_recovery"}.issubset(series.columns):
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["temperature_c"], series["area_proj_recovery"], label=route_titles["C"], linewidth=2.0, color="#16a34a")
        _draw_route_temperature_markers("C")
        plt.xlabel("温度（℃）")
        plt.ylabel("恢复比例")
        plt.title(f"{route_titles['C']}温度曲线")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "route_c_recovery_vs_temperature.png", dpi=160)
        plt.close(fig)

    if {"temperature_c", "direction_recovery"}.issubset(series.columns) and series["direction_recovery"].notna().any():
        fig = plt.figure(figsize=(8, 4.8))
        plt.plot(series["temperature_c"], series["direction_recovery"], label="方向法", linewidth=2.0, color="#c2410c")
        direction_result = result.direction_result or {}
        if direction_result.get("af95_c") is not None:
            plt.axvline(direction_result["af95_c"], color="tab:green", linestyle="--", label=f"95%恢复温度 {direction_result['af95_c']:.2f}℃")
        if direction_result.get("aftan_c") is not None:
            plt.axvline(direction_result["aftan_c"], color="tab:red", linestyle="--", label=f"切线法温度 {direction_result['aftan_c']:.2f}℃")
        angle_deg = direction_result.get("angle_deg")
        title = "方向法温度曲线" if angle_deg is None else f"方向法温度曲线 ({float(angle_deg):.1f}°)"
        plt.xlabel("温度（℃）")
        plt.ylabel("恢复比例")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "direction_recovery_vs_temperature.png", dpi=160)
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
        component_bridge_kernel=int(raw.get("component_bridge_kernel", 11)),
        min_component_area=int(raw.get("min_component_area", 80)),
        anchor_top_band_px=int(raw.get("anchor_top_band_px", 24)),
        route_a_tip_cluster_radius_px=float(raw.get("route_a_tip_cluster_radius_px", 6.0)),
        route_b_endpoint_extension_scale=float(raw.get("route_b_endpoint_extension_scale", 0.0)),
        route_b_cap_inset_scale=float(raw.get("route_b_cap_inset_scale", 0.08)),
        fit_bin_px=float(raw.get("fit_bin_px", 3.0)),
        fit_path_fraction=float(raw.get("fit_path_fraction", 0.72)),
        fit_path_fraction_min=float(raw.get("fit_path_fraction_min", 0.42)),
        fit_curvature_threshold_ratio=float(raw.get("fit_curvature_threshold_ratio", 0.28)),
        fit_margin_prefer_quadratic=float(raw.get("fit_margin_prefer_quadratic", 0.05)),
        anchor_prior_xy=tuple(raw.get("anchor_prior_xy")) if raw.get("anchor_prior_xy") is not None else None,
        anchor_prior_weight=float(raw.get("anchor_prior_weight", 0.0)),
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
    PREVIEWS_ROOT.mkdir(parents=True, exist_ok=True)
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
                frame_stride INTEGER NOT NULL DEFAULT 1,
                direction_angle_deg REAL,
                direction_metric_enabled INTEGER NOT NULL DEFAULT 0,
                actual_mode TEXT,
                formal_metric_label TEXT,
                formal_gate_reason TEXT,
                af95_c REAL,
                aftan_c REAL,
                original_frame_count INTEGER,
                analyzed_frame_count INTEGER,
                annotated_video_filename TEXT,
                video_filename TEXT NOT NULL,
                temperature_filename TEXT,
                run_dir TEXT NOT NULL,
                error_text TEXT
            )
            """
        )
        existing_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(runs)").fetchall()
        }
        if "frame_stride" not in existing_columns:
            conn.execute("ALTER TABLE runs ADD COLUMN frame_stride INTEGER NOT NULL DEFAULT 1")
        if "direction_angle_deg" not in existing_columns:
            conn.execute("ALTER TABLE runs ADD COLUMN direction_angle_deg REAL")
        if "direction_metric_enabled" not in existing_columns:
            conn.execute("ALTER TABLE runs ADD COLUMN direction_metric_enabled INTEGER NOT NULL DEFAULT 0")
        if "original_frame_count" not in existing_columns:
            conn.execute("ALTER TABLE runs ADD COLUMN original_frame_count INTEGER")
        if "analyzed_frame_count" not in existing_columns:
            conn.execute("ALTER TABLE runs ADD COLUMN analyzed_frame_count INTEGER")
        if "annotated_video_filename" not in existing_columns:
            conn.execute("ALTER TABLE runs ADD COLUMN annotated_video_filename TEXT")
        conn.commit()


def _cleanup_preview_cache(*, max_age_hours: float = 24.0) -> None:
    PREVIEWS_ROOT.mkdir(parents=True, exist_ok=True)
    cutoff_ts = datetime.now(timezone.utc).timestamp() - max_age_hours * 3600.0
    for path in PREVIEWS_ROOT.iterdir():
        try:
            if path.stat().st_mtime < cutoff_ts:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
        except OSError:
            continue


def _transcode_preview_video(source_path: Path, output_path: Path) -> Path:
    return transcode_video_for_browser(source_path, output_path, max_width=960, max_fps=15.0)


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


def _delete_run_record(run_id: str) -> None:
    with _connect_db() as conn:
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        conn.commit()


def _list_runs(limit: int) -> list[sqlite3.Row]:
    with _connect_db() as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY datetime(created_at) DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return list(rows)


def _delete_run_assets(run_id: str, run_dir: Path) -> None:
    fallback_dir = (RUNS_ROOT / run_id).resolve()
    try:
        resolved_dir = run_dir.resolve()
    except OSError:
        resolved_dir = fallback_dir
    try:
        resolved_dir.relative_to(RUNS_ROOT.resolve())
    except ValueError:
        resolved_dir = fallback_dir
    if resolved_dir.exists():
        shutil.rmtree(resolved_dir)


async def _save_upload(upload: UploadFile, path: Path) -> None:
    with path.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    await upload.close()


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


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
