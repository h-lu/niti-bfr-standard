from __future__ import annotations

from typing import Any

import pandas as pd

from .pipeline import BRAIDED_METRIC_ALIAS_TO_KEY, WIRE_ROUTE_ALIAS_TO_KEY

ROUTE_ALIAS_ORDER = ("A", "B", "C")
ROUTE_RESULTS_SCHEMA_VERSION = 4
OBJECT_RESULT_SCHEMA_VERSION = 2


def route_alias_for_metric_key(preset: str | None, metric_key: str | None) -> str | None:
    if metric_key is None:
        return None
    if str(preset or "").startswith("braided"):
        for alias, candidate in BRAIDED_METRIC_ALIAS_TO_KEY.items():
            if candidate == metric_key:
                return alias
        return None
    for alias, candidate in WIRE_ROUTE_ALIAS_TO_KEY.items():
        if candidate == metric_key:
            return alias
    return None


def canonical_object_result_fields(
    *,
    preset: str | None,
    reportability_status: str | None,
    formal_metric_key: str | None,
    formal_gate_reason: str | None,
    provisional_metric_key: str | None,
    af95_c: float | None,
    aftan_c: float | None,
    provisional_af95_c: float | None,
    provisional_aftan_c: float | None,
) -> dict[str, Any]:
    recommended_metric_key = formal_metric_key or provisional_metric_key
    return {
        "object_result_schema_version": OBJECT_RESULT_SCHEMA_VERSION,
        "object_reportability_status": reportability_status,
        "object_formal_metric_key": formal_metric_key,
        "object_formal_route_alias": route_alias_for_metric_key(preset, formal_metric_key),
        "object_formal_gate_reason": formal_gate_reason,
        "object_formal_af95_c": af95_c,
        "object_formal_aftan_c": aftan_c,
        "object_provisional_metric_key": provisional_metric_key,
        "object_provisional_route_alias": route_alias_for_metric_key(preset, provisional_metric_key),
        "object_provisional_af95_c": provisional_af95_c,
        "object_provisional_aftan_c": provisional_aftan_c,
        "object_recommended_metric_key": recommended_metric_key,
        "object_recommended_route_alias": route_alias_for_metric_key(preset, recommended_metric_key),
    }


def route_results_by_alias(route_results: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    by_alias: dict[str, dict[str, Any]] = {}
    for entry in route_results or []:
        alias = entry.get("alias")
        if isinstance(alias, str) and alias:
            by_alias[alias] = dict(entry)
    return by_alias


def flatten_route_results(route_results: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for entry in route_results or []:
        flat = {key: value for key, value in dict(entry).items() if key != "route_qc_summary"}
        qc_summary = entry.get("route_qc_summary") or {}
        if isinstance(qc_summary, dict):
            for key, value in qc_summary.items():
                if key == "stability_metric" and isinstance(value, dict):
                    flat["qc_stability_metric_key"] = value.get("key")
                    flat["qc_stability_metric_value"] = value.get("value")
                    continue
                flat[f"qc_{key}"] = value
        flattened.append(flat)
    return flattened


def route_results_dataframe(route_results: list[dict[str, Any]] | None) -> pd.DataFrame:
    return pd.DataFrame(flatten_route_results(route_results))
