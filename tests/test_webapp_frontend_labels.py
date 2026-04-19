from __future__ import annotations

import unittest

from niti_bfr.webapp import SAMPLE_RUNS, _mode_label, _preset_label, _run_result_hint


class WebappFrontendLabelTests(unittest.TestCase):
    def test_braided_sample_is_exposed(self) -> None:
        self.assertIn("braided_synthetic_quicklook", SAMPLE_RUNS)
        self.assertEqual(SAMPLE_RUNS["braided_synthetic_quicklook"]["preset"], "braided_demo")
        self.assertEqual(SAMPLE_RUNS["braided_synthetic_quicklook"]["requested_mode"], "quicklook")

    def test_labels_are_human_readable(self) -> None:
        self.assertEqual(_preset_label("braided_like"), "braided-like")
        self.assertEqual(_mode_label("formal_af"), "formal Af")

    def test_result_hint_shows_formal_downgrade(self) -> None:
        run = {
            "status": "completed",
            "requested_mode": "formal_af",
            "actual_mode": "quicklook",
            "formal_gate_reason": "body_mask_attachment_leak_fraction",
        }
        hint = _run_result_hint(run)
        self.assertIn("formal Af 未放行", hint)
        self.assertIn("quicklook", hint)

    def test_result_hint_shows_formal_success(self) -> None:
        run = {
            "status": "completed",
            "requested_mode": "formal_af",
            "actual_mode": "formal_af",
            "formal_gate_reason": None,
        }
        self.assertEqual(_run_result_hint(run), "formal Af 已放行。")


if __name__ == "__main__":
    unittest.main()
