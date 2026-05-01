import os
import random
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy.stats import rankdata

from kinship import Kinship
from population import Population

from individual_models.utility_based import utilityBasedAI
from individual_models.rule_based import ruleBasedAI
from individual_models.q_learning import qLearningAI

from territory import TerritoryMap


def run_simulation(diameter=20, subordinate_benefit=0.2, age_fitness_dict=None, life_history_fitness_dict=None, habitat_quality_dict=None, epsilon = 0.3):
    # CREATE DATASETS

    output_path = "output/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename_start = "data_"

    # tracks reward for individuals
    fitness_df = {}

    # tracks mean_kinship
    kinship_df = []

    # tracks individuals
    ind_df = []

    # tracks territories that exist each generation
    territory_df = []

    # for plotting territory
    colour_dict = {0: 0}

    # GET HABITAT QUALITY MAP

    cousin_island = Image.open("map/habitat_quality_small.png")

    # convert to greyscale
    cousin_island = cousin_island.convert("L")

    # convert to np array
    cousin_island = np.array(cousin_island)

    # INITIALISE SIMULATION

    # ploting years
    # the territory map will be plotted every ... years
    plot_years = 100

    #defines the age : fitness values, aka if age is higher the fitness is lower, this is to simulate senescence. Must be between 0 and 1, with 1 being the maximum fitness. The age of 11 represents the maximum age an individual can reach, and must have a fitness of 0.0
    if age_fitness_dict is None:
        age_fitness_dict = {
            0: 1.0,
            1: 1.0,
            2: 1.0,
            3: 1.0,
            4: 1.0,
            5: 1.0,
            6: 1.0,
            7: 0.9,
            8: 0.7,
            9: 0.3,
            10: 0.1,
            11: 0.0,
        }
    # defines the life history : fitness values, with primary having the highest fitness and floaters the lowest. Must be between 0 and 1, with 1 being the maximum fitness
    if life_history_fitness_dict is None:
        life_history_fitness_dict = {
            # must be <= 1 and > 0
            "fledgling": 1.0,
            "primary": 1.0,  # primary higher than subordinate
            "subordinate": 1.0,  # subordinate higher than floater
            "floater": 1.0,
        }
    # This maps the grayscale pixel colors of the provided Cousin Island map image to relative territory qualities (high, medium, low, or ocean)
    if habitat_quality_dict is None:
        habitat_quality_dict = {
            0: 1.4,  # high
            127: 1.2,  # medium
            195: 1,  # low
            255: 0,  # must always be 0, represents the ocean
        }

    # parameters
    carrying_capacity = 300 # number of individuals 
    init_pop_size = carrying_capacity 
    year, years = 0, 30
    init_sex_ratio = 0.5
    min_kinship = 0.1
    min_quality = 3 # minimum quality of a territory, determining the minimum carrying capacity. must be greater than 3


    # population    
    sex = ["female"]*init_pop_size
    n_males = int(init_pop_size*init_sex_ratio)
    sex[0:n_males] = ["male"] * n_males
    inds = list(range(init_pop_size))

    pop = Population(inds, sex)
    pop_dict = pop.get_dict()

    # kinship
    kinship = Kinship(pop, min_kinship)
    kinship.update()

    
    # territory

    quality_map = np.tile(0.0, cousin_island.shape).astype(np.float16())

    for colour, quality in habitat_quality_dict.items():
        quality_map[cousin_island == colour] = quality

    total_quality = np.sum(quality_map)
    quality_map = np.divide(quality_map, total_quality / carrying_capacity)

    territory_map = TerritoryMap(pop, quality_map, diameter, min_quality)
    territory_dict = territory_map.get_territories()

    for ind, sex in zip(inds, sex):
        fitness_df[ind] = {"ind": ind,
                           "sex": sex,
                           "year": 0,
                           "fitness": 0}

    #declaring which model to use
    '''
    individual_ai = ruleBasedAI(
                    pop=pop,
                    territory_map=territory_map,
                    kinship=kinship,
                    start_year=year,
                    min_kinship = min_kinship
                    )
    '''
    '''
    individual_ai = utilityBasedAI(
                    pop=pop,
                    territory_map=territory_map,
                    kinship=kinship,
                    min_kinship=min_kinship,
                    year=year,
                    diameter=diameter,
                    min_quality=min_quality,
                    )
    '''
    individual_ai = qLearningAI(
                    pop=pop,
                    territory_map=territory_map,
                    kinship=kinship,
                    min_kinship=min_kinship,
                    year=year,
                    diameter=diameter,
                    min_quality=min_quality,
                    epsilon=epsilon
                )

    #______________________________________________________________________________

    # RUN SIMULATION

    surviving = True

    while year <= years and surviving:

        territory_map.set_year(year)


    #______________________________________________________________________________

    # ACTION

        territory_map.reset_territory_competitions()
        territory_map.sync_territories(pop.get_inds())

        for ind in pop.get_inds():

            actions = pop.get_actions(ind)
            sex = pop[ind]["sex"]

            individual_ai._set_year(year)

            territory ,center, action = individual_ai.action(ind)

            # print(f"action taken for individual: {ind} was {action} at center {center} for territory {territory}")
            if action == "disperse":
                pop.update_life_history(ind, "floater")

            elif action == "request_subordinate" and territory is not None and territory in territory_dict:   
                territory_map.request_subordinate(ind, territory)
                # adds the indiviudal to a list of competing individuals for a primary position in a territory

            elif action == "compete_primary" and territory is not None and territory in territory_dict:
                territory_map.compete_primary(ind, sex, territory)
                # adds the indiviudal to a list of competing individuals for a primary position in a territory

            elif action == "establish_territory":
                success = territory_map.create_territory(ind, center) 
                # success contains a boolean value indicating if the creation of the territory was succesful
                # a loop could be created to attempt creating a different territory if an attempt was unsuccesful due to the location
                # be careful as creating a territory may not always be possbile!            

        # update territory map
        territory_map.update() # update territory map

    #______________________________________________________________________________

    # PRIMARY AND SUBORDINATE COMPETITION

        territory_map.sync_territories(pop.get_inds())

        unclaimed_territories = []

        for territory in territory_dict:

            # territory quality        
            quality = territory_map[territory]["quality"]

            territory_map[territory]["fledglings"] = []

            primary_male = territory_map[territory]["primary_male"]
            primary_female = territory_map[territory]["primary_female"]
            subordinates = territory_map[territory]["subordinates"]

            # if primary male has died, males compete for primary position
            if primary_male is None:
                territory_map.decide_primary(territory, "male")

            # if primary female has died, females compete for primary position
            if primary_female is None:
                territory_map.decide_primary(territory, "female")
                territory_map.sync_territories(pop.get_inds())
            primary_male = territory_map[territory]["primary_male"]
            primary_female = territory_map[territory]["primary_female"]

            # if territory has neither a primary male nor a priamry female
            if primary_female is None and primary_male is None:
                # remove territory 
                unclaimed_territories.append(territory)

            # if territory has both a primary male and primary female
            elif (not primary_female is None) and (not primary_male is None):

                evicted_subordinates = []
                new_subordinates = []            

                #======================================================================
                # EVICT SUBORDINATES 
                # INDIVIDUAL AI MODEL REPLACES THIS SECTION
                # subordinates may be evicted by either the primary male or primary female
                # this will likely be influenced by the territory quality, current number of subordinates, and relatedness

                for ind in territory_map[territory]["subordinates"]:
                    eviction = individual_ai.evict_subordinate_male_primary(ind)
                    if eviction:
                        evicted_subordinates.append(ind)            


                # ACCEPT NEW SUBORDINATES 
                # INDIVIDUAL AI MODEL REPLACES THIS SECTION
                # primary_male and primary_female choose if individual can assist, becoming a subordinate
                # this will likely be influenced by the territory quality, current number of subordinates, and relatedness

                for ind in territory_map[territory]["subordinate_request"]:
                    accept_subordinate = individual_ai.acccept_subordinate(ind)
                    if accept_subordinate:
                        new_subordinates.append(ind)

                #======================================================================

                # evict chosen subordinates
                for ind in evicted_subordinates:
                    territory_map.remove_subordinate(ind)

                # add chosen subordinates
                for ind in new_subordinates:
                    territory_map.add_subordinate(ind, territory)

        # disperse fledglings who unsuccesfully attempted to become a subordinate 
        for ind in pop.get_inds():
            if pop[ind]["life_history"] == "fledgling":
                pop.update_life_history(ind, "floater")

        for territory in unclaimed_territories:
            territory_map.remove_territory(territory)

    #______________________________________________________________________________

    # REPRODUCTION

        territory_map.sync_territories(pop.get_inds())

        for territory in territory_dict:

            primary_male = territory_map[territory]["primary_male"]
            primary_female = territory_map[territory]["primary_female"]
            subordinates = territory_map[territory]["subordinates"]
            quality = territory_map[territory]["quality"]
            # if territory has both a primary male and primary female
            if (not primary_female is None) and (not primary_male is None):

                # get female subordinates
                female_subordinates = []
                for subordinate in subordinates:
                    if pop[subordinate]["sex"] == "female":
                        female_subordinates.append(subordinate)

                # list of all females who will produce offspring 
                reproducing_females = [primary_female]

                #======================================================================
                # ALLOW SUBORDINATE REPRODUCTION
                # INDIVIDUAL AI MODEL REPLACES THIS SECTION    
                # primary_male and primary_female choose if subordinate female can reproduce (both must approve)
                # reproducing_females must contain a list of all females in the territory who will reproduce 

                for subordinate in female_subordinates:
                    reproduce = individual_ai.acccept_subordinate_reproduction(subordinate)
                    if reproduce:
                        reproducing_females.append(subordinate)

                #======================================================================

                # get number of subordinates 
                number_of_subordinates = len(subordinates)

                # REPRODUCTION
                for female in reproducing_females:
                
                    #print(len(reproducing_females))
                    new_ind = max(pop.get_inds())+1

                    offspring_sex = random.choice(("male", "female"))                

                    pop.add(father=primary_male, 
                            mother=female, 
                            ind=new_ind, 
                            sex=offspring_sex, 
                            fitness = 1.0, 
                            territory=territory, 
                            year = year, 
                            quality = quality, 
                            num_subordinates=number_of_subordinates)

    #______________________________________________________________________________

    # SURVIVAL

        # sync territory
        territory_map.sync_territories(pop.get_inds())

        start_pop_size = pop.pop_size()

        num_floaters = 0
        num_territory_inds = 0

        # track fitness
        territory_fitness = [] # inds within territories 
        floater_fitness = [] # floaters 

        for ind in list(pop.get_inds()):

            age = year - pop[ind]["year"]
            life_history = pop[ind]["life_history"]

            # reset fitness
            pop[ind]["fitness"] = 1.0

            # get number of floaters
            if life_history == "floater":
                num_floaters += 1

            else:
                num_territory_inds += 1

            # fledgling fitness benefit from number of subordinates 
            if life_history == "fledgling":

                territory = pop[ind]["territory"]

                num_subordinates = len(territory_map[territory]["subordinates"])

                pop[ind]["fitness"] *= 1 + (subordinate_benefit * num_subordinates)

            # age fitness
            pop[ind]["fitness"] *= age_fitness_dict[age]

            # life history fitness 
            pop[ind]["fitness"] *= life_history_fitness_dict[life_history]

            if life_history == "floater":
                # append ind fitness to pop_fitness for floaters
                floater_fitness.append(pop[ind]["fitness"])
            else:      
                # append ind fitness to pop_fitness for non-floaters
                territory_fitness.append(pop[ind]["fitness"])

        # calculate mean_fitness 
        mean_territory_fitness = np.mean(territory_fitness)
        mean_floater_fitness = np.mean(floater_fitness)

        territory_fitness = []
        for ind in list(pop.get_inds()):
            if pop[ind]["life_history"] != "floater":
                # guard if mean teritory fitness = 0
                if mean_territory_fitness > 0:

                    territory = pop[ind]["territory"]
                    territory_carrying_capacity = territory_map[territory]["quality"]
                    territory_pop_size = territory_map.count_inds(territory)
                    floaters_in_territory = num_floaters * (territory_carrying_capacity / carrying_capacity)

                    # calculate local density dependent selection 
                    # guard to prevent NaN
                    safe_floater = mean_floater_fitness if (not np.isnan(mean_floater_fitness)) else 0.0
                    fitness_scaling = territory_carrying_capacity / ((territory_pop_size * mean_territory_fitness) + (floaters_in_territory * safe_floater))


                    # update fitness 
                    pop[ind]["fitness"] *= fitness_scaling
                    pop[ind]["fitness"] = min(pop[ind]["fitness"], 1.0)
                    pop[ind]["fitness"] = max(pop[ind]["fitness"], 0.0)

                    # append to fitness list
                    territory_fitness.append(pop[ind]["fitness"])

        # update mean_territory_fitness after density dependent fitness scaling 
        mean_territory_fitness = np.mean(territory_fitness)

        if np.isnan(mean_territory_fitness):
            mean_territory_fitness = 0.0

        # estimate number of individuals living in territories who will survive 
        surviving_territory_inds = num_territory_inds * mean_territory_fitness

        for ind in list(pop.get_inds()):
            if pop[ind]["life_history"] == "floater":
                # guard to skip scaling if no floaters
                if num_floaters > 0 and mean_floater_fitness > 0:
                # calculate global density dependent selection 
                    fitness_scaling = (carrying_capacity - surviving_territory_inds) / (num_floaters * mean_floater_fitness)

                    # update fitness 
                    pop[ind]["fitness"] *= fitness_scaling
                    pop[ind]["fitness"] = min(pop[ind]["fitness"], 1.0)
                    pop[ind]["fitness"] = max(pop[ind]["fitness"], 0.0)

        # survival
        for ind in list(pop.get_inds()):
            fitness = pop[ind]["fitness"]

            if random.choices((True, False), weights=(1-fitness, fitness))[0]:            
                pop.remove(ind)

        if len(pop.get_inds()) == 0 or len(territory_map.territory_dict) == 0:
            surviving = False
    
    #______________________________________________________________________________

    # RECORD DATA

        # plot territory map
        if year % plot_years == 0:

            # get map
            image = territory_map.territory_map.copy()

            # rank data
            image = rankdata(image, method="dense")

            # reshape
            image = image.reshape(territory_map.territory_map.shape)

            # plot
            plt.figure(figsize=(4,4))
            plt.imshow(image, cmap=sns.color_palette("cubehelix", as_cmap=True), origin="upper", interpolation="nearest")
            plt.title("year:" + str(year))
            plt.savefig(os.path.join(output_path, "territory_map_year_" + str(year) + ".png"))
            plt.close()

        # update kinship
        kinship.update()
        # gets each individuals fitness from previous year to help with per year change in Q-learning reward calcs
        prev_fitness = {ind: fitness_df.get(ind, {"fitness": 0.0})["fitness"] for ind in pop.get_inds()}

        # add new individuals to fitness_df
        for ind in pop.get_inds():
            if pop[ind]["life_history"] == "fledgling":

                fitness_df[ind] = {"ind": ind,
                                   "sex": pop[ind]["sex"],
                                   "year": pop[ind]["year"],
                                   "fitness": 0}
        
            
        # update new individuals
        for ind in pop.get_inds():
            if pop[ind]["life_history"] == "fledgling":

                # dataset of relatives with greater kinship than the minimum threshold
                relatives = kinship.matrix[ind][kinship.matrix[ind] >= min_kinship]

                for i in relatives.index:
                    fitness_df[i]["fitness"] += relatives.loc[i].copy()
        #---------------------------------
        # calculating q-learning reward
        #---------------------------------
        for ind in pop.get_inds():
            # calculate all bonuses to reward calculations
            # life history bonus
            lh   = pop[ind]["life_history"]
            base = pop[ind]["fitness"]          

            lh_bonus = {
                        "primary":     3.0,
                        "subordinate": 1.0,
                        "fledgling":   0.3,
                        "floater":    -1.5,
                    }.get(lh, 0.0)
            # parental success bonus
            parental_success_bonus = 0.0
            if lh == "primary":
                offspring_list = pop[ind].get("offspring", [])
                for kid_id in offspring_list:
                    if kid_id in pop.get_inds(): # If the child is still alive
                        # Reward the parent for the child survival fitness
                        parental_success_bonus += (pop[kid_id]["fitness"] * 0.5)
            # sex bonus
            ind_sex = pop[ind]["sex"]
            sex_bonus = 1.5 if (ind_sex == "female" and lh in ("primary", "subordinate")) else 0.0
            
            
            # attempt at competing bonus
            attempt_bonus = 0.0
            if ind in individual_ai.current_decisions:
                _, attempted_action = individual_ai.current_decisions[ind]  # peek without popping
                if attempted_action in ("compete_primary", "request_subordinate", "establish_territory"):
                    attempt_bonus = 0.5
                    
            # change in fitness from last year 
            kinship_delta = (fitness_df.get(ind, {"fitness": 0.0})["fitness"] - prev_fitness.get(ind, 0.0))
            #reward calculation
            reward = (base + 
              lh_bonus + 
              sex_bonus + 
              attempt_bonus + 
              parental_success_bonus + 
              (0.5 * kinship_delta))
            # updating q values
            individual_ai.update_q_values(ind, reward)


       
        mean_k = float(np.mean(kinship.matrix)) if kinship.matrix.size > 0 else 0.0
        kinship_df.append({"year": year, "mean_kinship": mean_k})

        

        # tracks inds within the population
        for ind in pop.get_inds():
            ind_df.append({"year": year,
                           "ind": ind,
                           "age": year - pop[ind]["year"],
                           "sex": pop[ind]["sex"],
                           "life_history": pop[ind]["life_history"],
                           "territory": pop[ind]["territory"]})


        # tracks territories that exist each generation 
        for territory in territory_map.territory_dict:
            territory_df.append({"year": year,
                                 "territory": territory,
                                 "quality": territory_map[territory]["quality"],
                                 "num_subordinates": len(territory_map[territory]["subordinates"]),
                                 "num_fledglings": len(territory_map[territory]["fledglings"]),
                                 "size": territory_map[territory]["size"]})


    #______________________________________________________________________________

        print("year:", year, "  population size:", len(pop.get_inds()), "  territory count:", len(territory_map.territory_dict))

        if len(pop.get_inds()) == 0 or len(territory_map.territory_dict) == 0:
            surviving = False
            print("Population died at year", year, "with", len(pop.get_inds()), "individuals", "and", len(territory_map.territory_dict), "territories")

        year += 1
    # end of episode q learning and table update
    individual_ai.end_of_episode_update()
    individual_ai.save_q_table()
        
        
    
           
        
         


    #______________________________________________________________________________

    # SAVE DATASETS

    for df, name in zip([fitness_df.values(), ind_df, territory_df], ["fitness.csv", "population.csv", "territory.csv"]):
        print("writing", name, "to", output_path)
        records = list(df)
        pd.DataFrame.from_records(records).to_csv(os.path.join(output_path, name), index=False)


    def _json_default_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


    territory_snapshot = {
        territory_id: {
            key: value
            for key, value in territory_info.items()
            if key != "distance_map"
        }
        for territory_id, territory_info in territory_map.get_territories().items()
    }

    territory_snapshot_path = os.path.join(output_path, "territory_dict_no_distance_map.json")
    with open(territory_snapshot_path, "w", encoding="utf-8") as f:
        json.dump(territory_snapshot, f, indent=2, default=_json_default_serializer)

    print("writing territory_dict_no_distance_map.json to", output_path)

    # habitat quality map
    plt.figure(figsize=(8, 6))
    sns.heatmap(quality_map)
    plt.close()
    
    # calculates average inclusive fitness for all individuals 
    mean_inclusive_fitness = np.mean([v["fitness"] for v in fitness_df.values()])
    
    
    
    return mean_inclusive_fitness  
#-------------------------------------
# Q-learning Loop
#-------------------------------------

# runs for n_episode times saving and loading current and previous q-tables respectively
# epsilon decays
if __name__ == "__main__":
    
    n_episodes = 20
    episode_fitness = []
    current_epsilon = 0.3 

    for episode in range(n_episodes):
        print(f"\n=== EPISODE {episode + 1}/{n_episodes} ===")
        mean_fitness = run_simulation(epsilon=current_epsilon)
        episode_fitness.append(mean_fitness)
        
        # linear decay
        current_epsilon = max(0.1, 0.3 - (episode * (0.2 / n_episodes)))
        print(f"Mean inclusive fitness: {mean_fitness:.4f}, epsilon: {current_epsilon:.4f}")
        

    # Plotting per episode fitness with 10 episode rolling average 
    window = 10
    rolling_mean = pd.Series(episode_fitness).rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_episodes + 1), episode_fitness, marker="o", linewidth=1, 
            alpha=0.4, color="steelblue", label="Per-episode fitness")
    plt.plot(range(1, n_episodes + 1), rolling_mean, linewidth=2.5, 
            color="steelblue", label=f"{window}-episode rolling mean")
    plt.axhline(y=np.mean(episode_fitness[-10:]), color="red", linestyle="--",
                label=f"Last 10 mean: {np.mean(episode_fitness[-10:]):.3f}")
    plt.xlabel("Episode")
    plt.ylabel("Mean Inclusive Fitness")
    plt.title("Inclusive Fitness per Episode — Q-Learning Agent")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/inclusive_fitness_per_episode.png", dpi=150)
    plt.close()
