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

def mean_hlp():
    return

def per_teri_hlp():
    return

def get_result():
    terr = pd.read_csv('territory.csv')
    pop = pd.read_csv('population.py')
    fit = pd.read_csv('fitness.py')
    teri_counts = teri_counts()
    mean_grp = mean_grp()
    anual_adl_teri = anual_adl_teri()
    frst_yr_surv = frst_yr_surv()
    pop_size = pop_size()
    hpl_eff = hpl_eff()
    adlt_svvl = adlt_svvl()
    mean_hlp = mean_hlp()
    per_teri_hlp = per_teri_hlp()
    return [teri_counts, mean_grp, anual_adl_teri, frst_yr_surv, pop_size, hpl_eff, adlt_svvl, mean_hlp, per_teri_hlp]


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
        "Territory counts": {"minimize": False},
        "Mean group size by territory quality": {"minimize": False},
        "Annual adults survival by territory quality": {"minimize": False},
        "First year survival": {"minimize": False},
        "Population size": {"minimize": False},
        "Helper effect on yearling production": {"minimize": False},
        "Adult annual survival": {"minimize": False},
        "Mean helpers per territory": {"minimize": False},
        "percent of territories with helpers": {"minimize": False},
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
                "Territory counts": result[0],
                "Mean group size by territory quality": result[1],
                "Annual adults survival by territory quality": result[2],
                "First year survival": result[3],
                "Population size": result[4],
                "Helper effect on yearling production": result[5],
                "Adult annual survival": result[6],
                "Mean helpers per territory": result[7],
                "percent of territories with helpers": result[8],
            },
        )
    except Exception as e:
        client.mark_trial_failed(trial_index=trial_index)
        raise

pareto = client.get_pareto_optimal_parameters()
client.save_to_json_file("experiment.json")