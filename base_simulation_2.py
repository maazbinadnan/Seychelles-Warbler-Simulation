# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
from PIL import Image 

#______________________________________________________________________________

# POPULATION CLASS

class Population():
    
    def __init__(self, inds, sex):

        self.pop_dict = {}
        
        for i in range(len(inds)):
            self.pop_dict[inds[i]] = {
                    "father": 0,
                    "mother": 0,
                    "offspring": [],
                    "sex": sex[i],
                    "territory": None,
                    "life_history": "floater", 
                    "fitness": 1.0,
                    "year": 0 
                }
            
    # return indivindual dict when ind is called
    def __getitem__(self, ind):
        return self.pop_dict[ind]
    
    # get number of indivinduals in population
    def pop_size(self):
        return len(self.pop_dict)
    
    # get all inds
    def get_inds(self):
        return self.pop_dict.keys()
    
    def get_dict(self):
        return self.pop_dict
    
    # add indiviudals to tree
    def add(self, father, mother, ind, sex, fitness, territory, year, num_subordinates, quality):
        self.pop_dict[father]["offspring"].append(ind) 
        self.pop_dict[mother]["offspring"].append(ind) 
        self.pop_dict[ind] = {
            "father": father,
            "mother": mother,
            "offspring": [],
            "sex": sex,
            "territory": territory,
            "life_history": "fledgling", 
            "num_subordinates": num_subordinates,
            "quality": quality,
            "fitness": fitness,
            "year": year 
            }

    # remove individual from population
    def remove(self, ind):
        self.pop_dict.pop(ind)
            
    def update_life_history(self, ind, life_history, territory=None):
        self.pop_dict[ind]["territory"] = territory
        self.pop_dict[ind]["life_history"] = life_history
        
    def get_actions(self, ind):
        
        sex = self.pop_dict[ind]["sex"]
        life_history = self.pop_dict[ind]["life_history"]
        
        if sex == "male":
                
            if life_history == "fledgling":               
                actions = ["request_subordinate", "disperse"]
                
            elif life_history == "subordinate":
                actions = ["compete_primary", "disperse", "nothing"]
                    
            elif life_history == "floater":
                actions = ["compete_primary", "request_subordinate", "establish_territory", "nothing"]
           
            else: # primary
                actions = ["nothing"]
                
        else: # female 
                
            if life_history == "fledgling":               
                actions = ["request_subordinate", "disperse"]
                
            elif life_history == "subordinate":
                actions = ["compete_primary", "disperse", "nothing"]
                    
            elif life_history == "floater":
                actions = ["compete_primary", "request_subordinate", "nothing"]
                
            else: # primary
                actions = ["nothing"]
        
        return actions
        
#______________________________________________________________________________

# KINSHIP CLASS

class Kinship():
    def __init__(self, pop, min_kinship):
        inds = pop.get_inds()
        pop_size = pop.pop_size()
        self.pop = pop
        initial = np.tile(0.05, (pop_size,pop_size))
        np.fill_diagonal(initial, 1.0)
        self.matrix = pd.DataFrame(initial, columns=(inds), index=(inds), dtype="float16")
    
        self.min_kinship = min_kinship
    
    def update(self):

        # remove outdated indiviudals 
        self.remove_outdated()
        
        # get lists of individuals
        old_inds = list(self.matrix.columns)
        new_inds = list(set(self.pop.get_inds()).difference(set(old_inds)))
        all_inds = old_inds + new_inds
        size = len(all_inds)
        
        # crate empty dataframe
        new_matrix = pd.DataFrame(np.tile(0.05, (size,size)), columns=(all_inds), index=(all_inds))
        
        # add values for old_inds
        new_matrix.loc[old_inds, old_inds] = self.matrix.values

        # calculate kinship of new cohort 
        for ind in new_inds:
            
            mother = self.pop[ind]["mother"]
            father = self.pop[ind]["father"]

            # kinship 
            new_matrix.loc[:, ind] = np.multiply(0.5, np.add(new_matrix[mother].values, new_matrix[father].values)) # column
            new_matrix.loc[ind] = new_matrix.loc[:, ind] # row
            new_matrix.loc[ind, ind] = 1.0 
                
        # update matrix
        self.matrix = new_matrix.round(4)

    def return_df(self):
        return self.matrix
    
    def calculate_relatedness(self, ind1, ind2):
        return self.matrix[ind1, ind2]        
        
    def remove_outdated(self):
        inds = self.matrix.columns
        current_inds = self.pop.get_inds()
        common_inds = list(set(inds) & set(current_inds))
        
        # list all parents 
        parents = set()
        for ind in pop.get_inds():
            parents.add(pop[ind]["mother"])
            parents.add(pop[ind]["father"])
            
        # for each individual in the matrix
        for ind in inds:
            
            # if indiviudal is not currently in the population
            if not ind in current_inds:
                
                # if individual is not the mother or father of any living indiviual
                if not ind in parents:
                
                    # get kinship values for individual
                    ind_kinship = self.matrix[ind][self.matrix[ind] < 1.0]
                    ind_kinship = ind_kinship.loc[common_inds]
                    
                    # if it has a maximum kinship of less than the minimum to living individuals
                    if max(ind_kinship) < self.min_kinship:
                        
                        # remove the row and column of the individual
                        self.matrix.drop(columns=ind, inplace=True)
                        self.matrix.drop(index=ind, inplace=True)

#______________________________________________________________________________

# TERRITORY CLASS
     
class TerritoryMap():
    def __init__(self, pop, habitat_quality, diameter, min_quality):
        self.habitat_quality = habitat_quality
        self.pop = pop
        self.dims = habitat_quality.shape
        self.territory_map = None
        self.diameter = diameter
        self.min_quality = min_quality
        self.territory_dict = {}        
    
    def __getitem__(self, territory):
        return self.territory_dict[territory]

    def get_territories(self):
        return self.territory_dict
    
    def create_territory(self, primary_male, center):

        # track success of territory creation
        success = False
        
        centers = []
        
        for territory in self.territory_dict:
            centers.append(self.territory_dict[territory]["center"])
                    
        # check if this is the first territory 
        if len(self.territory_dict) > 0:
            distances = np.linalg.norm(np.array(centers) - center, axis=1) # euclidian distances
            territory = max(self.territory_dict.keys())+1
        else:
            # first territory 
            distances = np.array((self.diameter))
            territory = 1
        
        if np.min(distances) >= self.diameter/4: 
                        
            # check territory size
            new_territory = {
                "territory": territory,
                "center": center
                }
            
            if self.update(new_territory)["quality"] >= self.min_quality:                 

                # add territory
                self.territory_dict[new_territory["territory"]] = {
                    "primary_male": primary_male,
                    "primary_male_competition": [],
                    "primary_female": None,
                    "primary_female_competition": [],
                    "center": center,
                    "subordinates": [], 
                    "subordinate_request": [],
                    "fledglings": [],
                    "size": None,
                    "quality": None,
                    "distance_map": None
                    }
                
                self.pop.update_life_history(primary_male, "primary", new_territory["territory"])
                
                # territory creation was succeful
                success = True
                
                self.update()
        
        return success
    
    def update(self, new_territory = None):

        # if new_territory == None, this update is considered as a test, casuing the function to return the size and quality of the updated territory rather than update the class variables
        centers = []
        territories = []
        distance_maps = [] 
        
        # get inds with teritories
        if len(self.territory_dict) > 0:     
            for territory in self.territory_dict.keys():
                centers.append(self.territory_dict[territory]["center"])
                territories.append(territory)
                distance_maps.append(self.territory_dict[territory]["distance_map"])
        
        if new_territory != None:
            territories.append(new_territory["territory"])
            centers.append(new_territory["center"])
            distance_maps.append(None)
            
        # create array
        proximity = np.tile(0, (self.dims[0], self.dims[1], len(territories)+1))

        coordinates = np.stack(np.indices(self.dims), axis=-1).reshape(-1,2)

        for center, i, distance_map, territory in zip(centers, range(len(centers)), distance_maps, territories):
            if distance_map is None:
                new_distance_map = (self.diameter/2) - np.linalg.norm(coordinates - center, axis=1).reshape(self.dims)
                proximity[:, :, i+1] = new_distance_map
                if not new_territory is None:
                    if territory != new_territory["territory"]:
                        self.territory_dict[territory]["distance_map"] = new_distance_map.copy()
            else: 
                proximity[:, :, i+1] = self.territory_dict[territory]["distance_map"].copy()

        # get maxima for each x,y value across z axis
        max_proximity = proximity.max(axis=2, keepdims=True)

        # get mask for all x,y values equal to the maxima that are not equal to 0
        max_mask = np.logical_and(proximity == max_proximity, np.repeat((max_proximity != 0), proximity.shape[2], axis=2))
        
        # select teritory claims by randomly assigning numbers to all coordinates equal to the x,y maxima
        territory_map = np.argmax(np.where(max_mask, np.random.random(max_mask.shape), -1), axis=2)
        
        # convert to territory 
        territories = np.array([0]+territories)  
        territory_map = territories[territory_map]    

        if new_territory != None:
            return {
                "size": np.sum(territory_map == new_territory["territory"]),
                "quality": np.sum((territory_map == new_territory["territory"]) * territory_map),
                "territory_map": territory_map
                }

        else:
            
            self.territory_map = territory_map
            
            for territory in self.territory_dict.keys():
                self.territory_dict[territory]["size"] = np.sum(self.territory_map == territory)
                self.territory_dict[territory]["quality"] = np.sum((self.territory_map == territory) * self.habitat_quality)
                self.territory_map = territory_map
                
            self.check_territories() # remove territories without the required habitat quality

    def check_territories(self):
        removing = []
        for territory in self.territory_dict.keys():
            if not self.territory_dict[territory]["quality"] >= self.min_quality: 
                removing.append(territory)
            
        for territory in removing:
            self.remove_territory(territory)

    def remove_territory(self, territory):
        self.territory_dict.pop(territory)
        
    def request_subordinate(self, ind, territory):
        self.territory_dict[territory]["subordinate_request"].append(ind)

    def add_subordinate(self, ind, territory):
        self.pop.update_life_history(ind, "subordinate", territory)

    def remove_subordinate(self, ind):
        self.pop.update_life_history(ind, "floater")

    def compete_primary(self, ind, sex, territory):
        if sex == "male":
            self.territory_dict[territory]["primary_male_competition"].append(ind)
        else: # female
            self.territory_dict[territory]["primary_female_competition"].append(ind)
            
    def decide_primary(self, territory, sex):
        
        new_primary = None
        
        if sex == "male":
            if len(self.territory_dict[territory]["primary_male_competition"]) > 0:
                # age based competition, competing male with max age wins
                ages = []
                for ind in self.territory_dict[territory]["primary_male_competition"]:
                    ages.append(year - pop[ind]["year"])    
                
                new_primary = self.territory_dict[territory]["primary_male_competition"][np.argmax(ages)]
            
        else: # female
            if len(self.territory_dict[territory]["primary_female_competition"]) > 0:
                # the winning female is chosen uniformly randomly between competing females
                new_primary = random.choice(self.territory_dict[territory]["primary_female_competition"])
            
        # if a new primary is chosen
        if not new_primary is None:

            # update life history and territory of individual
            pop.update_life_history(new_primary, "primary", territory)
    
    def count_inds(self, territory):
        count = 0
        
        if self.territory_dict[territory]["primary_male"] is not None:
            count += 1

        if self.territory_dict[territory]["primary_female"] is not None:
            count += 1
        
        count += len(self.territory_dict[territory]["subordinates"])
        count += len(self.territory_dict[territory]["fledglings"])
        
        return count
    
    def sync_territories(self, inds):
                
        self.reset_territory_inds()
        
        for ind in inds:
                        
            # get individual territory information from pop
            territory = self.pop[ind]["territory"]
            sex = self.pop[ind]["sex"]
            life_history = self.pop[ind]["life_history"] 
            
            # if territory exists
            if self.pop[ind]["territory"] in self.territory_dict.keys():
                
                territory = self.territory_dict[self.pop[ind]["territory"]]

                if life_history == "fledgling":
                    territory["fledglings"].append(ind)
                
                if life_history == "subordinate":
                    territory["subordinates"].append(ind)
                
                elif life_history == "primary":
                    if sex == "male":
                        territory["primary_male"] = ind
                    else:
                        territory["primary_female"] = ind

            else:
                
                # set ind territory to None
                self.pop[ind]["territory"] = None
                
                # become floater
                self.pop[ind]["life_history"] = "floater"
                
    def reset_territory_competitions(self):
        for territory in self.territory_dict:
            self.territory_dict[territory]["primary_male_competition"] = []
            self.territory_dict[territory]["primary_female_competition"] = []
            self.territory_dict[territory]["subordinate_request"] = []

    def reset_territory_inds(self):
        for territory in self.territory_dict:
            self.territory_dict[territory]["primary_male"] = None
            self.territory_dict[territory]["primary_female"] = None
            self.territory_dict[territory]["subordinates"] = []
            self.territory_dict[territory]["fledglings"] = []          

#______________________________________________________________________________

# CREATE DATASETS

output_path = "output/"
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

#______________________________________________________________________________

# GET HABITAT QUALITY MAP

cousin_island = Image.open("map/habitat_quality_small.png")

# convert to greyscale
cousin_island =  cousin_island.convert("L")

# convert to np array
cousin_island = np.array(cousin_island)

#______________________________________________________________________________

# INITIALISE SIMULATION

# ploting years
# the territory map will be plotted every ... years
plot_years = 5

# changable parameters 
diameter = 20 # max diameter of territories. Must allow for at least min_quality to be possible
subordinate_benefit = 0.2 # 1 + (subordinate_benefit * number of subordinates)

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

life_history_fitness_dict = { # must be <= 1 and > 0
    "fledgling": 1.0, 
    "primary": 1.0, # primary higher than subordinate
    "subordinate": 1.0, # subordinate higher than floater
    "floater": 0.01
    }

habitat_quality_dict = {
    0: 1.4, # high
    127: 1.2, # medium
    195: 1, # low 
    255: 0 # must always be 0, represents the ocean
    }

# parameters
carrying_capacity = 300 # number of individuals 
init_pop_size = carrying_capacity 
year, years = 0, 50
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
    
#______________________________________________________________________________

# RUN SIMULATION

surviving = True

while year <= years and surviving:
        

#______________________________________________________________________________

# ACTION

    territory_map.reset_territory_competitions()
    territory_map.sync_territories(pop.get_inds())

    for ind in pop.get_inds():
                                
        actions = pop.get_actions(ind)
        
        #======================================================================
        # INDIVIDUAL AI MODEL REPLACES THIS SECTION         
        
        # AI model may have to select a territory or center to complete action
        territories = list(territory_dict.keys())
        center = (np.random.randint(0,quality_map.shape[0]), np.random.randint(0,quality_map.shape[1]))
        
        # if no territories exist
        if len(territories) == 0:
            while True:
                try:
                    actions.remove("compete_primary")
                except ValueError:
                    break
            while True:
                try:
                    actions.remove("request_subordinate")
                except ValueError:
                    break
        else:
            territory = np.random.choice(territories)


        if pop[ind]["life_history"] == "floater" and pop[ind]["sex"] == "male":
            action = "establish_territory"
            
        # if fledgling, request to become subordinate
        if pop[ind]["life_history"] == "fledgling":
            action = "request_subordinate"
            if pop[ind]["territory"] in territories:
                territory = pop[ind]["territory"]
            
        # if subordinate, compete for primary position
        if pop[ind]["life_history"] == "subordinate":
            action = "compete_primary"
            if pop[ind]["territory"] in territories:
                territory = pop[ind]["territory"]

        # randomly select action        
        else:
            action = random.choice(actions) # AI action model would go here 
       
        #======================================================================
        
        sex = pop[ind]["sex"]
        
        if action == "disperse":
            pop.update_life_history(ind, "floater")
            
        elif action == "request_subordinate":   
            territory_map.request_subordinate(ind, territory)
            # adds the indiviudal to a list of competing individuals for a primary position in a territory
            
        elif action == "compete_primary":
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
                male_choice = random.choices((True,False), weights=(0.1, 0.9))
                female_choice = random.choices((True,False), weights=(0.1, 0.9))
                if male_choice or female_choice:
                    evicted_subordinates.append(ind)            
            
            
            # ACCEPT NEW SUBORDINATES 
            # INDIVIDUAL AI MODEL REPLACES THIS SECTION
            # primary_male and primary_female choose if individual can assist, becoming a subordinate
            # this will likely be influenced by the territory quality, current number of subordinates, and relatedness
        
            for ind in territory_map[territory]["subordinate_request"]:
                male_choice = random.choices((True,False), weights=(0.9, 0.1))
                female_choice = random.choices((True,False), weights=(0.9, 0.1))
                if male_choice and female_choice:
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
                male_choice = random.choices((True,False), weights=(1.0, 0.0))
                female_choice = random.choices((True,False), weights=(1.0, 0.0))
                if female_choice and male_choice:
                    reproducing_females.append(subordinate)
                    
            #======================================================================
                    
            # get number of subordinates 
            number_of_subordinates = len(subordinates)

            # REPRODUCTION
            for female in reproducing_females:
            
                #print(len(reproducing_females))
                new_ind = max(pop.get_inds())+1
                                
                sex = random.choice(("male", "female"))                

                pop.add(primary_male, 
                        female, 
                        new_ind, 
                        sex, 
                        1.0, 
                        territory, 
                        year, 
                        quality, 
                        number_of_subordinates)

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
        plt.show()
        
    # update kinship
    kinship.update()
    
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
                
    # tracks mean_kinship
    kinship_df.append({"year": year, 
                      "mean_kinship": np.mean(kinship.matrix)})

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

print("Simulation ended!")
        
#______________________________________________________________________________

# SAVE DATASETS

for df, name in zip([fitness_df.values(), ind_df, territory_df], ["fitness.csv", "population.csv", "territory.csv"]):
    print("writing", name, "to", output_path)
    pd.DataFrame.from_dict(df).to_csv(output_path + name, index = False)

# habitat quality map
plt.figure(figsize=(8, 6))
sns.heatmap(quality_map)
plt.show()



