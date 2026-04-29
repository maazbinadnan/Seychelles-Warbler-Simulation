from main import run_simulation
from ax.api.client import Client
import pandas as pd
import numpy as np

#The functions between the dashed lines are to be defined
#These functions should return the corresponding number as result, you can change arguments as how you use the function in get_result()
#----------------------------------------------------------------------------------------------
def teri_counts():# territory counts
    return

def mean_grp():#mean group size by territory quality
    return

def anual_adl_teri():# annual adult survival by territory quality
    return

def frst_yr_surv(df):#first year survival, this function has been done
    return len(df[(df['ind'] == 0) & (df['fitness'] > 0)])

def pop_size(df):# population size, finished
    max_age = df['age'].max()
    return len(df[df['age'] == max_age])

def hpl_eff():#helper effect on yearling production
    return

def adlt_svvl():# adult annual survival
    return

def mean_hlp():#mean helpers per territory
    return

def per_teri_hlp():# % territories with helper
    return

def get_result():# this is the function generating the final output metric
    # the first 3 lines are to read the result files
    terr = pd.read_csv('territory.csv')# read territory.csv
    pop = pd.read_csv('population.csv')# read population.csv
    fit = pd.read_csv('fitness.csv')# read fitness.csv
    # calculate the scores, every variable should be a number score
    territory_counts = teri_counts()
    mean_grp_size = mean_grp()
    survival_teri_quality = anual_adl_teri()
    first_survial = frst_yr_surv(fit) # finished
    population_size = pop_size(pop)#finished
    helper_effect = hpl_eff()
    adult_survival = adlt_svvl()
    mean_helpers = mean_hlp()
    percent_terri = per_teri_hlp()

    return [territory_counts, mean_grp_size, survival_teri_quality, first_survial, population_size, helper_effect, adult_survival, mean_helpers, percent_terri]

#----------------------------------------------------------------------------------------------------------------
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