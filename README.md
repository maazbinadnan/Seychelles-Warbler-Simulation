# Seychelles Warbler Simulation

## About

This is an agent-based simulation of cooperative breeding in the Seychelles Warbler (*Acrocephalus sechellensis*). It explores how kinship, territory quality, and different AI decision strategies influence helping behaviour and territory formation.

---

## Overview

The simulation models individual birds across a spatial habitat map (Cousin Island) over multiple years. Each bird holds one of four life-history states — **fledgling**, **subordinate**, **floater**, or **primary** — and makes decisions each year based on kinship, territory quality, and ecological constraints. Primary pairs also make decisions about evicting subordinates, accepting new helpers, and allowing subordinate reproduction.

---

## Project Structure

```
main.py                  # Simulation entry point — runs the full loop
population.py            # Population class — individual life histories and fitness
kinship.py               # Kinship class — pairwise relatedness tracking
territory.py             # TerritoryMap class — spatial territory allocation
multiple_test_runs.py    # Runs the simulation N times and saves each to its own folder
fine_tuning_all.py       # Parameter optimisation generates Pareto-front(requires ax-platform)
individual_models/
    rule_based.py        # ruleBasedAI — deterministic decision engine
    utility_based.py     # utilityBasedAI — utility-function decision engine
    genetic_algorithm.py # GeneticController — evolved decision weights (active default)
    q_learning.py        # qLearningAI — reinforcement-learning decision engine
map/                     # Habitat quality map (Cousin Island, greyscale PNG + CSV)
output/                  # CSV output files from the most recent single run
multiple_test_runs_output/
    plot_mean_fitness.py     # Plot mean fitness across runs
    plot_territory_count.py  # Plot territory count across runs
    plot_territory_stats.py  # Plot territory quality statistics across runs
    run_1/ run_2/ run_3/     # Per-run CSV output
legacy/                  # Archived original monolithic script
```

---

## Setup

**Requirements:** Python 3.10+

```bash
# 1. Create and activate a virtual environment (Windows)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

> `ax-platform` is **not** installed by default. It is only needed for `fine_tuning_all.py`.
> To use it: `pip install ax-platform==1.2.4`

---

## Running the Simulation

### Single run

```bash
python main.py
```

Output CSVs are written to `output/`:

| File | Contents |
|---|---|
| `fitness.csv` | Per-individual fitness scores by year |
| `population.csv` | Life-history state of every individual each year |
| `territory.csv` | Territory size and quality metrics by year |
| `territory_dict_no_distance_map.json` | Full territory state snapshot |

Territory maps are also rendered and displayed as plots during the run (every 100 years by default).

---

### Multiple runs

```bash
python multiple_test_runs.py
```

Runs the simulation 3 times and saves each run to `multiple_test_runs_output/run_<n>/`. To change the number of runs, edit `RUN_COUNT` at the top of `multiple_test_runs.py`.

After the runs complete, generate plots from inside `multiple_test_runs_output/`:

```bash
python plot_mean_fitness.py
python plot_territory_count.py
python plot_territory_stats.py
```

---

### Switching the AI model

Open `main.py` and locate the model block (~line 160). Only one model should be active at a time — uncomment the one you want and comment out the others:

| Model | Class | Notes |
|---|---|---|
| Genetic Algorithm | `GeneticController` | **Active by default** |
| Rule-Based | `ruleBasedAI` | Deterministic decision rules |
| Utility-Based | `utilityBasedAI` | Weighted utility scoring |
| Q-Learning | `qLearningAI` | RL agent; `epsilon` controls exploration |

---

## Command-line Arguments

- **`--ai`**: Choose which AI model to run the simulation with. Usage:

```bash
python main.py --ai q_learning
python main.py --ai genetic_algorithm
```

- **Default**: `rule_based` (no flag required).

- **`q_learning` special behaviour**: running with `--ai q_learning` performs multiple episodes (controlled by `n_episodes` in `main.py`, default 20) with a decaying `epsilon` exploration rate. The run produces `output/inclusive_fitness_per_episode.png` and the Q-table is saved by the Q-learning agent.

- **Advanced / programmatic usage**: `main.py` exposes `run_simulation(...)` which can be called from other scripts to override internal parameters. Example:

```python
from main import run_simulation
run_simulation(diameter=30, subordinate_benefit=0.25, epsilon=0.2, output_path="output/custom/", ai_name="utility")
```

- **Common `run_simulation` kwargs**:
    - **`diameter`**: (int) maximum territory diameter — default 20
    - **`subordinate_benefit`**: (float) fitness multiplier per subordinate — default 0.2
    - **`epsilon`**: (float) exploration rate for Q-Learning — default 0.3
    - **`output_path`**: (str) folder to write CSVs and images — default `output/`
    - **`ai_name`**: (str) the AI to use when calling programmatically (`utility`, `q_learning`, `genetic_algorithm`, `rule_based`)


### Parameter optimisation (optional)

```bash
pip install ax-platform==1.2.4
python fine_tuning_all.py
```

Runs a  multi-objective Bayesian optimisation over key simulation parameters. Pareto front are written to `pareto.txt`, optimization logs are written to 'experiemnt.json', summaries are written to 'summary.json'.

---

## Key Parameters (`main.py` → `run_simulation`)

| Parameter | Default | Description |
|---|---|---|
| `carrying_capacity` | 300 | Maximum population size |
| `years` | 30 | Simulation length (years) |
| `min_kinship` | 0.1 | Minimum relatedness for kin-based tolerance |
| `min_quality` | 3 | Minimum territory quality to be viable |
| `diameter` | 20 | Maximum territory diameter (pixels) |
| `subordinate_benefit` | 0.2 | Fitness multiplier per subordinate helper |
| `epsilon` | 0.3 | Exploration rate (Q-Learning only) |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | 2.4.4 | Numerical arrays and map operations |
| `pandas` | 3.0.2 | CSV output and data analysis |
| `scipy` | 1.17.1 | Rank-based fitness scoring |
| `matplotlib` | 3.10.8 | Territory and fitness plotting |
| `seaborn` | 0.13.2 | Statistical plots |
| `Pillow` | 12.2.0 | Loading the habitat quality map image |
| `ax-platform` | 1.2.4 | Pareto optimisation *(optional)* |
