from main import run_simulation
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
import pandas as pd
import numpy as np
import os
#The functions between the dashed lines are to be defined
#These functions should return the corresponding number as result, you can change arguments as how you use the function in get_result()
#----------------------------------------------------------------------------------------------
def get_territory_quality(df):
    newdf = df.copy()
    newdf["avg_pxl_qlty"] = newdf["quality"]/newdf["size"]
    highCutoff = newdf["avg_pxl_qlty"].quantile(0.879)
    medCutoff = newdf["avg_pxl_qlty"].quantile(0.708)
    def classify(val):
        if val >= highCutoff:
            return "High"
        elif val >= medCutoff:
            return "Medium"
        else:
            return "Low"
    newdf["category"] = newdf["avg_pxl_qlty"].apply(classify)    
    return newdf

def teri_counts(df):# territory counts, yearly average
    return df.groupby("year")["territory"].count().mean()

def mean_grp(popdf, terrdf):#mean group size by territory quality
    #update the df to put the territories into categories
    newTerrDf = get_territory_quality(terrdf)
    merged = pd.merge(newTerrDf[["year", "territory", "category"]], popdf, on=["year", "territory"])
    grpCounts = merged.groupby(["year", "territory", "category"]).size().reset_index(name="birdCount")
    return grpCounts.groupby("category")["birdCount"].mean().to_dict()

def anual_adl_teri(popDf, terriDf):# annual adult survival by territory quality
    terrDf = get_territory_quality(terriDf)
    
    #tallies for the each territory qualities, [seen, survived]
    counts = {"Low": [0,0], "Medium": [0,0], "High": [0,0]}

    merged = pd.merge(popDf[["year", "ind", "age", "territory"]],
                      terrDf[["year", "territory", "category"]],
                      on=["year", "territory"])
    years = merged["year"].unique()
    for year in years[:-1]:
        currentAdults = merged[(merged["year"]==year) & (merged["age"]>=1)]
        nextYearInd = set(popDf[popDf["year"]==year+1]["ind"].unique())

        for _, warb in currentAdults.iterrows():
            quality = warb["category"]
            if quality in counts:
                counts[quality][0] += 1
                if warb["ind"] in nextYearInd:
                    counts[quality][1] += 1

    return {q: (tallies[1]/tallies[0] if tallies[0]>0 else 0) for q, tallies in counts.items()}

def frst_yr_surv(df):#first year survival, this function has been done
    return len(df[(df['ind'] == 0) & (df['fitness'] > 0)])

def pop_size(df):# population size, finished
    return df.groupby("year")["ind"].count().mean()

#def hpl_eff():#helper effect on yearling production
  #  return

def adlt_svvl(df):# adult annual survival
    survivalRates = []
    #loops through each year
    for year in df["year"].unique()[:-1]:
        #gets the ind of all adults in this year
        currentAdlt = df[(df["year"]==year) & (df["age"]>=1)]["ind"].unique()

        #skips if there are no adults, like the first year
        if len(currentAdlt) == 0:
            continue
        
        #gets the ind of all warbs next year, no repeats speeds up if
        nextYearAlive = set(df[df["year"]==year+1]["ind"].unique())
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
    terr = pd.read_csv('output/territory.csv')# read territory.csv
    pop = pd.read_csv('output/population.csv')# read population.csv
    fit = pd.read_csv('output/fitness.csv')# read fitness.csv
    # calculate the scores, every variable should be a number score
    territory_counts = teri_counts(terr)
    grp = mean_grp(pop, terr)
    grp_low = grp["Low"]
    grp_medium = grp["Medium"]
    grp_high = grp["High"]
    surv = anual_adl_teri(pop, terr)
    surv_low = surv["Low"]
    surv_medium = surv["Medium"]
    surv_high = surv["High"]
    first_survial = frst_yr_surv(fit) # finished
    population_size = pop_size(pop)#finished
    #helper_effect = hpl_eff()
    adult_survival = adlt_svvl(pop)
    mean_helpers = mean_hlp(terr)
    percent_terri = per_teri_hlp(terr) 

    return [territory_counts, grp_low, grp_medium, grp_high,surv_low, surv_medium, surv_high, first_survial, population_size, adult_survival, mean_helpers, percent_terri]

#----------------------------------------------------------------------------------------------------------------
client = Client()

client.configure_experiment(
    name="multi_objective",
    parameters=[
        RangeParameterConfig(name="diameter",            parameter_type="float", bounds=(1.0, 20.0)),
        RangeParameterConfig(name="subordinate_benefit", parameter_type="float", bounds=(0.0, 1.0)),
        *[RangeParameterConfig(name=f"age_{i}",          parameter_type="float", bounds=(0.1, 1.0)) for i in range(11)],
        RangeParameterConfig(name="lh_fledgling",        parameter_type="float", bounds=(0.1, 1.0)),
        RangeParameterConfig(name="lh_primary",          parameter_type="float", bounds=(0.1, 1.0)),
        RangeParameterConfig(name="lh_subordinate",      parameter_type="float", bounds=(0.1, 1.0)),
        RangeParameterConfig(name="lh_floater",          parameter_type="float", bounds=(0.01, 0.05)),
        RangeParameterConfig(name="hq_high",             parameter_type="float", bounds=(1.5, 2.0)),
        RangeParameterConfig(name="hq_medium",           parameter_type="float", bounds=(1.0, 1.5)),
        RangeParameterConfig(name="hq_low",              parameter_type="float", bounds=(0.5, 1.0)),
    ],
)
client.configure_optimization(
    objective=(
        "territory_counts, grp_low, grp_medium, grp_high, surv_low, surv_medium, surv_high, "
        "first_survival, population_size, adult_survival, "
        "mean_helpers, percent_terri"
    )
)

n_trials = 30
for i in range(n_trials):
    print(f'iteration: {i}')
    trials = client.get_next_trials(max_trials=1)

    for trial_index, parameters in trials.items():
        try:
            age_fitness_dict = {j: parameters[f"age_{j}"] for j in range(11)}
            age_fitness_dict[11] = 0.0

            life_history_fitness_dict = {
                "fledgling":   parameters["lh_fledgling"],
                "primary":     parameters["lh_primary"],
                "subordinate": parameters["lh_subordinate"],
                "floater":     parameters["lh_floater"],
            }

            habitat_quality_dict = {
                0:   parameters["hq_high"],
                127: parameters["hq_medium"],
                195: parameters["hq_low"],
                255: 0,
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
                    "territory_counts":   (result[0], 0.0),
                    "grp_low":            (result[1], 0.0),
                    "grp_medium":         (result[2], 0.0),
                    "grp_high":           (result[3], 0.0),
                    "surv_low":           (result[4], 0.0),
                    "surv_medium":        (result[5], 0.0),
                    "surv_high":          (result[6], 0.0),
                    "first_survival":     (result[7], 0.0),
                    "population_size":    (result[8], 0.0),
                    "adult_survival":     (result[9], 0.0),
                    "mean_helpers":       (result[10], 0.0),
                    "percent_terri":      (result[11], 0.0),
                },
            )
        except Exception as e:
            print(f"Trial {trial_index} failed: {e}")
            client.mark_trial_failed(trial_index=trial_index)
            n_trials += 1
            continue

pareto = client.get_pareto_frontier()
client.save_to_json_file("experiment.json")