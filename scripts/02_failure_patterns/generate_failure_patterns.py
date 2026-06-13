"""Generate the pre-defined communication-failure patterns used for testing."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import xlsxwriter


REPO_ROOT = Path(__file__).resolve().parents[2]
EPISODE_LENGTH = 946_656
SENSORS = ("T2", "T3", "T4", "T6")
DURATION_STEPS = {"12h": 144, "24h": 288, "48h": 576}
DURATION_SEED_OFFSETS = {"12h": 0, "24h": 100, "48h": 200}
N_PATTERNS = 10


def generate_failure_pattern(
    failure_probability: float,
    failure_duration: str,
    random_seed: int,
) -> list[int]:
    """Return one binary failure sequence for the complete 2001-2009 test period."""
    failure_steps = DURATION_STEPS[failure_duration]
    failure_pattern: list[int] = []
    failure_counter = 0

    np.random.seed(random_seed)
    for time_step in range(EPISODE_LENGTH):
        if time_step == 0:
            failure_pattern.append(0)
        elif failure_counter > 0:
            failure_counter -= 1
            failure_pattern.append(1)
        else:
            trigger = np.random.uniform(0, 1)
            if trigger <= failure_probability:
                failure_counter = failure_steps
            # A triggered disconnection starts at the next decision step.
            failure_pattern.append(0)

    return failure_pattern


def write_pattern(path: Path, pattern: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(str(path))
    worksheet = workbook.add_worksheet("sheet1")
    worksheet.write_column(0, 0, pattern)
    workbook.close()


def generate_all_patterns(failure_probability: float, output_root: Path) -> None:
    """Generate 10 independent patterns for every sensor-duration combination."""
    for sensor_index, sensor in enumerate(SENSORS):
        sensor_seed = sensor_index * 1000
        for duration in DURATION_STEPS:
            duration_seed = sensor_seed + DURATION_SEED_OFFSETS[duration]
            for pattern_id in range(N_PATTERNS):
                random_seed = duration_seed + pattern_id
                pattern = generate_failure_pattern(
                    failure_probability,
                    duration,
                    random_seed,
                )
                output_path = (
                    output_root
                    / f"probability {failure_probability}"
                    / f"{sensor}_{duration}_{pattern_id}.xlsx"
                )
                write_pattern(output_path, pattern)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate communication-failure patterns for manuscript tests."
    )
    parser.add_argument(
        "--probability",
        type=float,
        choices=(0.001, 0.01),
        required=True,
        help="Per-step disconnection triggering probability.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data_required" / "failure_patterns",
        help="Root folder for generated pattern workbooks.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()
    generate_all_patterns(args.probability, args.output_root)
    print(f"Generation time: {time.time() - start_time:.2f} s")
