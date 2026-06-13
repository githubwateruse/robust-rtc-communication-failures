"""Identify the best saved generation for each manuscript policy."""

from pathlib import Path
import re

import numpy as np
import xlrd


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICIES = ("SV0", "SV1", "SV2", "SV3", "SV4", "T2", "T3", "T4", "T6")


def generation_number(path):
    match = re.fullmatch(r"Gen_(\d+)\.xlsx", path.name)
    if match is None:
        raise ValueError(f"Unexpected generation filename: {path.name}")
    return int(match.group(1))


def identify_selected_generation(policy):
    run_dir = REPO_ROOT / "data_required" / "training_runs" / policy
    generation_files = sorted(run_dir.glob("Gen_*.xlsx"), key=generation_number)
    if not generation_files:
        raise FileNotFoundError(
            f"No generation files found in {run_dir}. Run training first or "
            "place the saved Gen_*.xlsx files in this directory."
        )

    fitness_values = []
    for generation_file in generation_files:
        workbook = xlrd.open_workbook(str(generation_file))
        sheet = workbook.sheet_by_index(0)
        fitness_values.append(-sheet.cell_value(0, 1))

    selected_position = int(np.argmin(fitness_values))
    selected_file = generation_files[selected_position]
    return generation_number(selected_file), fitness_values[selected_position]


if __name__ == "__main__":
    for policy_name in POLICIES:
        generation, fitness = identify_selected_generation(policy_name)
        print(f"{policy_name}: generation={generation}, fitness={fitness}")
