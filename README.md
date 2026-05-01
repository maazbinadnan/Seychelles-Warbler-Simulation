# Seychelles Warbler Simulation

An agent-based simulation of cooperative breeding dynamics in the Seychelles Warbler (*Acrocephalus sechellensis*), built for SCC-452. The model investigates kin selection, territory quality, and the evolution of subordinate helping behaviour using a rule-based AI decision engine.

---

## Overview

The simulation models individual birds across a spatial habitat map (Cousin Island) over multiple years. Each bird holds one of four life-history states — **fledgling**, **subordinate**, **floater**, or **primary** — and makes decisions each year based on kinship, territory quality, and ecological constraints. Primary pairs also make decisions about evicting subordinates, accepting new helpers, and allowing subordinate reproduction.

---

## Project Structure

```
main.py             # Simulation entry point — runs the full loop
population.py       # Population class — individual life histories and fitness
kinship.py          # Kinship class — pairwise relatedness tracking
territory.py        # TerritoryMap class — spatial territory allocation
rule_based.py       # ruleBasedAI class — decision engine for all bird roles
fine_tuning_all.py  # Parameters optimization - output pareto front into pareto.txt
map/                # Habitat quality map (Cousin Island, greyscale PNG)
output/             # CSV output files (fitness, population, territory)
legacy/             # Archived original monolithic script
```

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `carrying_capacity` | 300 | Maximum population size |
| `years` | 10 | Simulation length |
| `min_kinship` | 0.1 | Minimum relatedness for kin-based tolerance |
| `min_quality` | 3 | Minimum territory quality to be viable |
| `diameter` | 20 | Maximum territory diameter (pixels) |
| `subordinate_benefit` | 0.2 | Fitness multiplier per subordinate helper |

---

## Decision Rules (ruleBasedAI)

- **Fledgling**: females remain on high-quality territories; males disperse with moderate probability.
- **Subordinate**: leaves if unrelated to primary female, if old (age ≥ 8), or if a vacancy arises to challenge.
- **Floater**: competes for vacancies, attempts to establish new territories, or requests subordinate status.
- **Primary (evict)**: female evicts unrelated subordinates; male evicts on poor territories with multiple helpers.
- **Primary (accept)**: female accepts related candidates; male accepts related or high-quality territory candidates.
- **Primary (reproduction)**: subordinate female may only reproduce on high-quality territories and if related to the primary female.

---

## Running the Simulation

```bash
# Activate the virtual environment (Windows)
.venv\Scripts\Activate.ps1

# Run
python main.py
```

Output CSVs are written to `output/`. Territory maps are plotted every 5 years.

---

## Dependencies

- Python 3
- `numpy`, `pandas`, `scipy`
- `matplotlib`, `seaborn`
- `Pillow`