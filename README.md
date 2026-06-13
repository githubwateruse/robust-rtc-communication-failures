# Information-aware multi-policy control for urban drainage systems under communication failures

This repository provides the core implementation materials for the manuscript:

**Robust real-time urban drainage control under communication failures: an information-aware multi-policy control framework**

The repository contains the main computational workflow used in the study, including neural-network policy training, selected-policy identification, communication-failure pattern generation, single-policy testing, and multi-policy switching testing. The scripts were reorganized from the authors' working implementation to make the workflow easier to follow while preserving the core experimental logic used in the manuscript.

## Repository contents

* `SWMM_Astlingen/`: SWMM input models and rainfall input files.
* `scripts/01_training/`: neural-network policy training and selected-policy identification.
* `scripts/02_failure_patterns/`: communication-failure pattern generation.
* `scripts/03_single_policy_testing/`: single-policy testing under ideal and communication-failure conditions.
* `scripts/04_multi_policy_switching/`: multi-policy switching experiments under communication failures.
* `trained_policies/`: selected neural-network control policies used in the manuscript experiments.
* `data_required/`: folders for generated failure patterns, training workbooks, and simulation outputs.

## Requirements

The original experiments were conducted using Python 3.8.0. The main Python dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```text
# Python 3.8.0
numpy==1.23.4
pyswmm==1.4.0
xlrd==1.2.0
XlsxWriter==3.2.0
```

The package versions reflect the environment used for the core implementation scripts. Minor version adjustments may be required depending on the local SWMM, PySWMM, and operating-system configuration.

The scripts are research implementation programs rather than a packaged software library. Experiment settings, such as state-vector configuration, failure duration, failure probability, and failed sensor, are defined in the corresponding training or testing scripts.

## SWMM model and rainfall data

The repository includes the SWMM input models and the compact synthetic training rainfall files used for policy training.

The larger 2001–2009 testing rainfall files are not bundled in this repository. Before running the testing scripts, please prepare the four testing rainfall files listed in `SWMM_Astlingen/rainfall_testing/README.md` and place them in that directory.

The `.inp` files use repository-relative rainfall paths where possible. Users may still need to adjust local file paths according to their own SWMM/PySWMM setup.

## Selected policies

The repository provides the selected neural-network control policies used in the manuscript experiments. These policies were obtained from the original training workflow by selecting the best-performing saved generation for each state-vector configuration.

The selected policy files are provided directly in `trained_policies/`, so users can run the testing scripts without retraining the neural-network policies from scratch.

For long PySWMM-based training runs, the original workflow used continuation batches. In the manuscript experiments, the Python process was restarted after each 40-generation training segment, and training was then continued from the saved generation workbooks.

## Core workflow

### Step 1: Train neural-network policies

Set the state-vector configuration in:

```text
scripts/01_training/train_nn_policies.py
```

Then run the script for the desired policy configuration. The script saves generation workbooks under:

```text
data_required/training_runs/
```

To continue a previous run, place the saved generation workbooks in the corresponding policy folder and set `GENERATION_OFFSET` to the next generation number.

### Step 2: Identify selected policies

Run:

```bash
python scripts/01_training/identify_selected_policies.py
```

This script scans the available `Gen_*.xlsx` workbooks and applies the original fitness criterion to identify the best saved generation for each policy configuration.

### Step 3: Generate communication-failure patterns

Run:

```bash
python scripts/02_failure_patterns/generate_failure_patterns.py --probability 0.001
python scripts/02_failure_patterns/generate_failure_patterns.py --probability 0.01
```

For each failure probability, the script generates communication-failure patterns for the monitored sensors and failure durations used in the manuscript. The generated pattern IDs `0`–`9` correspond to the repeated communication-failure realizations used in the testing experiments.

### Step 4: Test single-policy control

Use the scripts in:

```text
scripts/03_single_policy_testing/
```

to evaluate selected policies under ideal conditions and under communication-failure scenarios.

### Step 5: Test multi-policy switching control

Use the scripts in:

```text
scripts/04_multi_policy_switching/
```

to evaluate the information-aware multi-policy switching strategies under communication-failure scenarios.

Simulation outputs are written below:

```text
data_required/simulation_outputs/
```

## Scope and reproducibility notes

This repository is intended to provide the core implementation logic and main computational workflow used in the manuscript. The original experiments were conducted in the authors' local Python/SWMM environment.

Although the scripts have been reorganized and machine-specific paths have been replaced with repository-relative paths where possible, users may still need to adjust local paths, SWMM model settings, rainfall input files, Python package versions, and PySWMM/SWMM toolkit settings according to their own computing environment.

The provided scripts are therefore best understood as research implementation materials for reproducing and understanding the main experimental workflow, rather than as a fully packaged software application.

## Acknowledgements

This repository uses the Astlingen SWMM benchmark model introduced by Sun et al. (2020):

Sun, C., Svensen, J. L., Borup, M., Puig, V., Cembrano, G., and Vezzaro, L. (2020). *An MPC-Enabled SWMM Implementation of the Astlingen RTC Benchmarking Network*. Water, 12(4), 1034. https://doi.org/10.3390/w12041034

The Astlingen SWMM model files are available from the open-toolbox repository:
https://github.com/open-toolbox/SWMM-Astlingen

We sincerely acknowledge the authors for developing and publicly providing the Astlingen SWMM benchmark model.
