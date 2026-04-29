from main import run_simulation
from ax.api.client import Client
import pandas as pd
import numpy as np

#The functions between the dashed lines are to be defined
#These functions should return the corresponding number as result, you can change arguments as how you use the function in get_result()
#----------------------------------------------------------------------------------------------
def teri_counts(df):# territory counts, yearly average
    return df.groupby("year")["territory"].count().mean()

def mean_grp(popdf, terrdf):#mean group size by territory quality
    merged = pd.merge(terrdf[["year", "territory", "quality"]], popdf, on=["year", "territory"])
    grpSizes = merged.groupby(["year", "territory", "quality"]).size().reset_index(name="size")
    results = {}
    for quality in [1.0,1.2,1.4]:
        results[quality] = grpSizes[grpSizes["quality"]==quality]["size"].mean()
    return results

def anual_adl_teri(popDf, terrDf):# annual adult survival by territory quality
    #tallies for the each territory qualities, [seen, survived]
    counts = {1.0: [0,0], 1.2: [0,0], 1.4: [0,0]}

    merged = pd.merge(popDf[["year", "ind", "age", "territory"]],
                      terrDf[["year", "territory", "quality"]],
                      on=["year", "territory"])
    years = merged["year"].unique()
    for year in years[:-1]:
        currentAdults = merged[(merged["year"]==year) & (merged["age"]>=1)]

        nextYearInd = merged[merged["year"]==year+1]["ind"].unique()

        for _, warb in currentAdults.iterrows():
            quality = warb["quality"]
            if quality in counts:
                counts[quality][0] += 1
                if warb["ind"] in nextYearInd:
                    counts[quality][1] += 1

    return {q: (tallies[1]/tallies[0] if tallies[0]>0 else 0) for q, tallies in counts.items()}

def frst_yr_surv(df):#first year survival, this function has been done
    return len(df[(df['ind'] == 0) & (df['fitness'] > 0)])

def pop_size(df):# population size, finished
    max_age = df['age'].max()
    return len(df[df['age'] == max_age])

def hpl_eff():#helper effect on yearling production
    return

def adlt_svvl(df):# adult annual survival
    survivalRates = []
    #loops through each year
    for year in df["year"].unique()[:-1]:
        #gets the ind of all adults in this year
        currentAdlt = df[(df["year"]==year) & (df["age"]>=1)]["ind"].unique()

        #skips if there are no adults, like the first year
        if currentAdlt == 0:
            continue
        
        #gets the ind of all warbs next year
        nextYearAlive = df[df["year"]==year+1]["ind"].unique()
        #counts all the adults in this year who are alive next year
        surviveCount = sum(1 for ind in currentAdlt if ind in nextYearAlive)

        survivalRates.append(surviveCount/len(currentAdlt) * 100)

    return np.mean(survivalRates)

def mean_hlp(df):#mean helpers per territory, takes the average for each year (like Brouwers 2012), and then averages over all the years
    averagePerYear = df.groupby("year")["num_subordinates"].mean()
    return averagePerYear.mean()

def per_teri_hlp(df):# % percentage of territories with helper for each year (like Brouwers 2012) 
    yearlyPct = df.groupby("year").apply(lambda x: (x["num_subordinates"]>0).sum()/len(x) * 100)
    return yearlyPct.mean()

def get_result():# this is the function generating the final output metric
    # the first 3 lines are to read the result files
    terr = pd.read_csv('territory.csv')# read territory.csv
    pop = pd.read_csv('population.csv')# read population.csv
    fit = pd.read_csv('fitness.csv')# read fitness.csv
    # calculate the scores, every variable should be a number score
    territory_counts = teri_counts(terr)
    mean_grp_size = mean_grp(pop, terr)
    survival_teri_quality = anual_adl_teri(pop, terr)
    first_survial = frst_yr_surv(fit) # finished
    population_size = pop_size(pop)#finished
    helper_effect = hpl_eff()
    adult_survival = adlt_svvl(pop)
    mean_helpers = mean_hlp(terr)
    percent_terri = per_teri_hlp(terr) 

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