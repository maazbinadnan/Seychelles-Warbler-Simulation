# -*- coding: utf-8 -*-

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy.stats import rankdata

from kinship import Kinship
from population import Population
from individual_models.rule_based import ruleBasedAI
from territory import TerritoryMap


def run_simulation():
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
    plot_years = 5

    # changable parameters
    diameter = 20  # max diameter of territories. Must allow for at least min_quality to be possible
    subordinate_benefit = 0.2  # 1 + (subordinate_benefit * number of subordinates)

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

    life_history_fitness_dict = {
        # must be <= 1 and > 0
        "fledgling": 1.0,
        "primary": 1.0,  # primary higher than subordinate
        "subordinate": 1.0,  # subordinate higher than floater
        "floater": 0.01,
    }

    habitat_quality_dict = {
        0: 1.4,  # high
        127: 1.2,  # medium
        195: 1,  # low
        255: 0,  # must always be 0, represents the ocean
    }

    # parameters
    carrying_capacity = 300  # number of individuals
    init_pop_size = carrying_capacity
    year, years = 0, 10
    init_sex_ratio = 0.5
    min_kinship = 0.1
    min_quality = 3  # minimum quality of a territory, determining the minimum carrying capacity. must be greater than 3

    # population
    sex = ["female"] * init_pop_size
    n_males = int(init_pop_size * init_sex_ratio)
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

    individual_ai = ruleBasedAI(
        pop=pop,
        territory_map=territory_map,
        kinship=kinship,
        min_kinship=min_kinship,
        year=year,
        diameter=diameter,
        min_quality=min_quality,
    )

    for ind, ind_sex in zip(inds, sex):
        fitness_df[ind] = {"ind": ind, "sex": ind_sex, "year": 0, "fitness": 0}

    # RUN SIMULATION

    surviving = True

    while year <= years and surviving:
        individual_ai.set_year(year)
        territory_map.set_year(year)

        # ACTION

        territory_map.reset_territory_competitions()
        territory_map.sync_territories(pop.get_inds())

        for ind in pop.get_inds():
            action, territory, center = individual_ai.decide(ind)

            ind_sex = pop[ind]["sex"]

            if action == "disperse":
                pop.update_life_history(ind, "floater")

            elif action == "request_subordinate" and territory in territory_dict:
                territory_map.request_subordinate(ind, territory)
                # adds the indiviudal to a list of competing individuals for a primary position in a territory

            elif action == "compete_primary" and territory in territory_dict:
                territory_map.compete_primary(ind, ind_sex, territory)
                # adds the indiviudal to a list of competing individuals for a primary position in a territory

            elif action == "establish_territory":
                success = territory_map.create_territory(ind, center)
                # success contains a boolean value indicating if the creation of the territory was succesful
                # a loop could be created to attempt creating a different territory if an attempt was unsuccesful due to the location
                # be careful as creating a territory may not always be possbile!

        # update territory map
        territory_map.update()  # update territory map

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

            # if territory has neither a primary male nor a priamry female
            if primary_female is None and primary_male is None:
                # remove territory
                unclaimed_territories.append(territory)

            # if territory has both a primary male and primary female
            elif (primary_female is not None) and (primary_male is not None):
                evicted_subordinates = []
                new_subordinates = []

                #======================================================================
                # EVICT SUBORDINATES
                for ind in territory_map[territory]["subordinates"]:
                    male_evicts, female_evicts = individual_ai.decide_evict_subordinate(
                        primary_male, primary_female, ind, territory
                    )
                    if male_evicts or female_evicts:
                        evicted_subordinates.append(ind)

                # ACCEPT NEW SUBORDINATES
                for ind in territory_map[territory]["subordinate_request"]:
                    male_accepts, female_accepts = individual_ai.decide_accept_subordinate(
                        primary_male, primary_female, ind, territory
                    )
                    if male_accepts and female_accepts:
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

        # REPRODUCTION

        territory_map.sync_territories(pop.get_inds())

        for territory in territory_dict:
            primary_male = territory_map[territory]["primary_male"]
            primary_female = territory_map[territory]["primary_female"]
            subordinates = territory_map[territory]["subordinates"]

            # if territory has both a primary male and primary female
            if (primary_female is not None) and (primary_male is not None):
                # get female subordinates
                female_subordinates = []
                for subordinate in subordinates:
                    if pop[subordinate]["sex"] == "female":
                        female_subordinates.append(subordinate)

                # list of all females who will produce offspring
                reproducing_females = [primary_female]

                #======================================================================
                # ALLOW SUBORDINATE REPRODUCTION
                for subordinate in female_subordinates:
                    male_allows, female_allows = individual_ai.decide_subordinate_reproduction(
                        primary_male, primary_female, subordinate, territory
                    )
                    if male_allows and female_allows:
                        reproducing_females.append(subordinate)

                #======================================================================

                # get number of subordinates
                number_of_subordinates = len(subordinates)

                # REPRODUCTION
                for female in reproducing_females:
                    new_ind = max(pop.get_inds()) + 1

                    ind_sex = random.choice(("male", "female"))

                    pop.add(
                        primary_male,
                        female,
                        new_ind,
                        ind_sex,
                        1.0,
                        territory,
                        year,
                        quality,
                        number_of_subordinates,
                    )

        # SURVIVAL

        # sync territory
        territory_map.sync_territories(pop.get_inds())

        start_pop_size = pop.pop_size()

        num_floaters = 0
        num_territory_inds = 0

        # track fitness
        territory_fitness = []  # inds within territories
        floater_fitness = []  # floaters

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
                territory = pop[ind]["territory"]
                territory_carrying_capacity = territory_map[territory]["quality"]
                territory_pop_size = territory_map.count_inds(territory)
                floaters_in_territory = num_floaters * (territory_carrying_capacity / carrying_capacity)

                # calculate local density dependent selection
                fitness_scaling = territory_carrying_capacity / ((territory_pop_size * mean_territory_fitness) + (floaters_in_territory * mean_floater_fitness))

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
                # calculate global density dependent selection
                fitness_scaling = (carrying_capacity - surviving_territory_inds) / (num_floaters * mean_floater_fitness)

                # update fitness
                pop[ind]["fitness"] *= fitness_scaling
                pop[ind]["fitness"] = min(pop[ind]["fitness"], 1.0)
                pop[ind]["fitness"] = max(pop[ind]["fitness"], 0.0)

        # survival
        for ind in list(pop.get_inds()):
            fitness = pop[ind]["fitness"]

            if random.choices((True, False), weights=(1 - fitness, fitness))[0]:
                pop.remove(ind)

        if len(pop.get_inds()) == 0 or len(territory_map.territory_dict) == 0:
            surviving = False

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
            plt.figure(figsize=(4, 4))
            plt.imshow(image, cmap=sns.color_palette("cubehelix", as_cmap=True), origin="upper", interpolation="nearest")
            plt.title("year:" + str(year))
            plt.show()

        # update kinship
        kinship.update()

        # add new individuals to fitness_df
        for ind in pop.get_inds():
            if pop[ind]["life_history"] == "fledgling":
                fitness_df[ind] = {"ind": ind, "sex": pop[ind]["sex"], "year": pop[ind]["year"], "fitness": 0}

        # update new individuals
        for ind in pop.get_inds():
            if pop[ind]["life_history"] == "fledgling":
                # dataset of relatives with greater kinship than the minimum threshold
                relatives = kinship.matrix[ind][kinship.matrix[ind] >= min_kinship]

                for i in relatives.index:
                    fitness_df[i]["fitness"] += relatives.loc[i].copy()

        # tracks mean_kinship
        kinship_df.append({"year": year, "mean_kinship": np.mean(kinship.matrix)})

        # tracks inds within the population
        for ind in pop.get_inds():
            ind_df.append(
                {
                    "year": year,
                    "ind": ind,
                    "age": year - pop[ind]["year"],
                    "sex": pop[ind]["sex"],
                    "life_history": pop[ind]["life_history"],
                    "territory": pop[ind]["territory"],
                }
            )

        # tracks territories that exist each generation
        for territory in territory_map.territory_dict:
            territory_df.append(
                {
                    "year": year,
                    "territory": territory,
                    "quality": territory_map[territory]["quality"],
                    "num_subordinates": len(territory_map[territory]["subordinates"]),
                    "num_fledglings": len(territory_map[territory]["fledglings"]),
                    "size": territory_map[territory]["size"],
                }
            )

        print("year:", year, "  population size:", len(pop.get_inds()), "  territory count:", len(territory_map.territory_dict))

        if len(pop.get_inds()) == 0 or len(territory_map.territory_dict) == 0:
            surviving = False
            print(
                "Population died at year",
                year,
                "with",
                len(pop.get_inds()),
                "individuals",
                "and",
                len(territory_map.territory_dict),
                "territories",
            )

        year += 1

    print("Simulation ended!")

    # SAVE DATASETS

    for df, name in zip([fitness_df.values(), ind_df, territory_df], ["fitness.csv", "population.csv", "territory.csv"]):
        print("writing", name, "to", output_path)
        pd.DataFrame.from_dict(df).to_csv(output_path + name, index=False)

    # habitat quality map
    plt.figure(figsize=(8, 6))
    sns.heatmap(quality_map)
    plt.show()


if __name__ == "__main__":
    run_simulation()
