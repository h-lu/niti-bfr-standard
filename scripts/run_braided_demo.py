from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from niti_bfr.braided_benchmark import run_braided_benchmark


def main() -> None:
    run = run_braided_benchmark("braided_demo", root=ROOT)
    print(f"{run.scenario.name} saved to {run.output_dir}")


if __name__ == "__main__":
    main()
