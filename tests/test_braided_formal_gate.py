from __future__ import annotations

import numpy as np
import pandas as pd

from niti_bfr.metrics import MetricEvaluation, RecoveryFit
from niti_bfr.pipeline import _augment_braided_qc_series, _formal_braided_af_gate


def _axis_report() -> MetricEvaluation:
    return MetricEvaluation(
        label="length_axis",
        increasing=False,
        fit=RecoveryFit(x_m=-210.0, x_a=-170.0, t0=47.0, width=2.0),
        af95_c=50.0,
        aftan_c=49.0,
        fit_rmse=0.5,
        monotonic_violation_fraction=0.0,
        dynamic_range=40.0,
    )


def _series_with_branch_qc(column: str, value: float) -> pd.DataFrame:
    n = 20
    recovery = np.concatenate([np.linspace(0.0, 0.84, n - 5), np.linspace(0.92, 1.0, 5)])
    return pd.DataFrame(
        {
            "temperature_c": np.linspace(30.0, 60.0, n),
            "length_axis_px": np.linspace(210.0, 170.0, n),
            "length_axis_recovery": recovery,
            "quality": np.full(n, 0.85),
            column: np.full(n, value),
        }
    )


def test_augment_braided_qc_series_preserves_branch_count_aliases() -> None:
    modern = _augment_braided_qc_series(pd.DataFrame({"branch_component_count_after_pruning": [1.0, 2.0]}))
    legacy = _augment_braided_qc_series(pd.DataFrame({"branch_count_after_pruning": [3.0, 4.0]}))

    assert modern["branch_count_after_pruning"].tolist() == [1.0, 2.0]
    assert legacy["branch_component_count_after_pruning"].tolist() == [3.0, 4.0]


def test_formal_braided_gate_rejects_high_branch_component_count() -> None:
    series = _series_with_branch_qc("branch_component_count_after_pruning", 5.0)

    allowed, reason = _formal_braided_af_gate(series, {"length_axis": _axis_report()})

    assert not allowed
    assert reason == "branch_component_count_after_pruning"


def test_formal_braided_gate_accepts_legacy_branch_count_alias() -> None:
    series = _series_with_branch_qc("branch_count_after_pruning", 1.0)

    allowed, reason = _formal_braided_af_gate(series, {"length_axis": _axis_report()})

    assert allowed
    assert reason is None


def test_formal_braided_gate_rejects_legacy_branch_count_alias_when_high() -> None:
    series = _series_with_branch_qc("branch_count_after_pruning", 5.0)

    allowed, reason = _formal_braided_af_gate(series, {"length_axis": _axis_report()})

    assert not allowed
    assert reason == "branch_component_count_after_pruning"
