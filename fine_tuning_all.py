from main import run_simulation
from ax.api.client import Client
import pandas as pd
import numpy as np

def teri_counts():
    return

def mean_grp():
    return

def anual_adl_teri():
    return

def frst_yr_surv():
    return

def pop_size():
    return

def hpl_eff():
    return

def adlt_svvl():
    return

def mean_hpl():
    return

def get_result():
    terr = pd.read_csv('territory.csv')
    pop = pd.read_csv('population.py')
    fit = pd.read_csv('fitness.py')


client = Client()

client.create_experiment(
    name="multi_objective",
    parameters=[
        {"name": "diameter", "type": "range", "bounds": [1.0, 5.0]},
        {"name": "subordinate_benefit", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "age_0", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "age_1", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "age_2", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "age_3", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "age_4", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "age_5", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "age_6", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "age_7", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "age_8", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "age_9", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "age_10", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "lh_fledgling", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "lh_primary", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "lh_subordinate", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "lh_floater", "type": "range", "bounds": [0.01, 0.01]},
        {"name": "hq_high", "type": "range", "bounds": [1.0, 2.0]},
        {"name": "hq_medium", "type": "range", "bounds": [1.0, 2.0]},
        {"name": "hq_low", "type": "range", "bounds": [0.5, 1.5]},
    ],
    objectives={
        "metric1": {"minimize": False},
        "metric2": {"minimize": True},
    },
)
n_trials = 30
for _ in range(n_trials):
    parameters, trial_index = client.get_next_trial()

    try:
        age_fitness_dict = {i: parameters[f"age_{i}"] for i in range(11)}
        age_fitness_dict[11] = 0.0

        life_history_fitness_dict = {
            "fledgling": parameters["lh_fledgling"],
            "primary": parameters["lh_primary"],
            "subordinate": parameters["lh_subordinate"],
            "floater": parameters["lh_floater"],
        }

        habitat_quality_dict = {
            0: parameters["hq_high"],
            127: parameters["hq_medium"],
            195: parameters["hq_low"],
            255: 0,  # ocean always 0
        }
        run_simulation(
            diameter=parameters["diameter"],
            subordinate_benefit=parameters["subordinate_benefit"],
            age_fitness_dict=age_fitness_dict,
            life_history_fitness_dict=life_history_fitness_dict,
            habitat_quality_dict=habitat_quality_dict,
        )
        result = get_result()

        client.complete_trial(
            trial_index=trial_index,
            raw_data={
                "metric1": result[0],
                "metric2": result[1],
            },
        )
    except Exception as e:
        client.mark_trial_failed(trial_index=trial_index)
        raise

pareto = client.get_pareto_optimal_parameters()
client.save_to_json_file("experiment.json")