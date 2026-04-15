# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# INITIALISE 

years = 100 # number of iterations of simulation

# simulation paramerers 
lifespan = 8
fecundity = 4
teritory_value = 0.1
K = 100

# ancestry
class Population():
    
    def __init__(self, inds, sexes):

        self.pop_dict = {}
        
        for i in range(len(inds)):
            self.pop_dict[inds[i]] = {
                    "father": 0,
                    "mother": 0,
                    "offspring": [],
                    "sex": sexes[i],
                    "center": [np.random.randint(0, 50),np.random.randint(0, 100)],
                    "diameter": 25,
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
    def add(self, father, mother, inds, sexes, fitness, year):
        for i in range(len(inds)):
            self.pop_dict[father]["offspring"].append(inds[i]) 
            self.pop_dict[mother]["offspring"].append(inds[i]) 
            self.pop_dict[inds[i]] = {
                    "father": father,
                    "mother": mother,
                    "offspring": [],
                    "sex": sexes[i],
                    "center": None,
                    "diameter": 25,
                    "fitness": fitness[i],
                    "year": year 
                }

    # remove dead indivinduals    
    def remove(self, inds):
        for ind in inds:
            #self.pop_dict[ind]
            self.pop_dict.pop(ind)
        

class Kinship():
    def __init__(self, pop_dict, min_kinship):
        inds = pop_dict.get_inds()
        pop_size = pop_dict.pop_size()
        self.pop_dict = pop_dict
        initial = np.tile(0.0, (pop_size,pop_size))
        np.fill_diagonal(initial, 1.0)
        self.matrix = pd.DataFrame(initial, columns=(inds), index=(inds))
    
        self.min_kinship = min_kinship
    
    def update(self):
        
        # remove outdated indiviudals 
        self.remove_outdated()
        
        # get lists of individuals
        old_inds = list(self.matrix.columns)
        new_inds = list(set(self.pop_dict.get_inds()).difference(set(old_inds)))
        all_inds = old_inds + new_inds
        size = len(all_inds)
        
        # crate empty dataframe
        new_matrix = pd.DataFrame(np.tile(0.0, (size,size)), columns=(all_inds), index=(all_inds))
        
        # add values for old_inds
        new_matrix.loc[old_inds, old_inds] = self.matrix.values

        # calculate kinship of new cohort 
        for ind in new_inds:
            
            mother = self.pop_dict[ind]["mother"]
            father = self.pop_dict[ind]["father"]

            # kinship 
            new_matrix.loc[:, ind] = (0.5 * new_matrix[mother]) + (0.5 * new_matrix[father]) # column
            new_matrix.loc[ind] = new_matrix.loc[:, ind] # row
            new_matrix.loc[ind, ind] = 1.0 
                
        # update matrix
        self.matrix = new_matrix
    
    def return_df(self):
        return self.matrix
        
    def remove_outdated(self):
        inds = self.matrix.columns
        current_inds = self.pop_dict.get_inds()
        
        # for each individual in the matrix
        for ind in inds:
            
            # if it is not currently in the population
            if not ind in current_inds:
                                
                # if it has a maximum kinship of less than the minimum
                #if max(self.matrix[ind][self.matrix[ind] != 1]) < self.min_kinship:
                if max(self.matrix[ind].drop(ind)) < self.min_kinship:
 
                    # remove the row and column of the individual
                    self.matrix.drop(columns=ind, inplace=True)
                    self.matrix.drop(index=ind, inplace=True)
   
     
class TerritoryMap():
    def __init__(self, pop_dict, grid):
        self.pop_dict = pop_dict
        self.dims = grid.shape
        self.territory = None
        self.distance = None

    def get_territories(self):
        return self.territory
    
    def get_distances(self):
        return self.distance
    
    def update(self, test):
        
        # get inds with teritories
        inds = []
                
        for ind in self.pop_dict.get_inds():           
            if self.pop_dict[ind]["center"] != None:
                inds.append(ind)
        
        # create array
        proximity = np.tile(0, (self.dims[0], self.dims[1], len(inds)+1))
        
        for ind, i in zip(inds, range(len(inds))):
            
            diameter = self.pop_dict[ind]["diameter"]

            coordinates = np.stack(np.indices(self.dims), axis=-1).reshape(-1,2)
            proximity[:, :, i+1] = (diameter/2) - np.linalg.norm(coordinates - self.pop_dict[ind]["center"], axis=1).reshape(self.dims)

        # get maxima for each x,y value across z axis
        max_proximity = proximity.max(axis=2)
          
        # proximity above 0 mask 
        zero_mask = np.repeat((max_proximity != 0)[:, :, None], proximity.shape[2], axis=2)

        # list all coorinates with the max proximity for each grid position what are not equal to 0  
        territorial_claims = np.argwhere(np.logical_and((proximity == max_proximity[:, :, None]), zero_mask))

        # randomise order of territorial_claims
        np.random.shuffle(territorial_claims)

        # select the first index of each unique x,y position
        index = np.unique(territorial_claims[:, :2], axis=0, return_index=True)[1]
        territorial_claims = territorial_claims[index] 

        # assign teritory values
        territory = np.tile(0, self.dims)
        territory[territorial_claims[:,0], territorial_claims[:,1]] = territorial_claims[:,2]      
                
        # convert to individual IDs
        inds = np.array([0]+inds)  
        
        territory = inds[territory]
        
        if test:
            return territory
        else:
            self.territory = territory





# https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/j.1365-2656.2006.01155.x
# (84%) and juveniles (61%) survival
# Local density had a negative effect on survival;
# birds living in larger groups had lower survival probabilities than those living in small groups. 
# Food availability did not affect survival.


# https://academic.oup.com/beheco/article/27/1/295/1744875
# either through female chocie or male - male competition, older males mate more succesfully 

# https://eprints.whiterose.ac.uk/id/eprint/119521/1/2016%20Komdeur%20chapter%20proofs.pdf

# https://www.nature.com/articles/s41467-019-09229-3
# longer lifespan with subordiantes
# late-life decline in survival is greatly reduced in female dominants when a helper is present. 
# the probability of having female, but not male, helpers increases with dominant female age. 
# 

# https://nsojournals.onlinelibrary.wiley.com/doi/full/10.1111/j.0908-8857.2008.04124.x
# sex biased dispersal, increased in females likely to avoid inbreeding 

# https://research-portal.uea.ac.uk/en/publications/investigating-inbreeding-avoidance-and-dispersal-in-a-contained-p/
# sex biased dispersal not caused by inbreeding avoidance

# https://www.nature.com/articles/358493a0
# habitat saturation and teritory size

# https://pmc.ncbi.nlm.nih.gov/articles/PMC1691204/
# breeding can occur in first year, although some remain as subordinates
# helping is affected by habitat saturation and variation in territory quality
# long-term benefits of helping are higher for daughters than for sons
# daughters are more likely to be helpers than males
# low-quality territories breeding pairs raising sons gain higher fitness benefits than by raising daughters, and vice versa on high-quality territories. 
# seychelles warblers lay a single-egg clutch
# Female breeders adaptively modify the sex of their single-egg clutches according to territory quality: male eggs on low quality and female eggs on high quality. 

# https://www.researchgate.net/publication/223978005_Helpers_at_the_Nest_Improve_Late-Life_Offspring_Performance_Evidence_from_a_Long-Term_Study_and_a_Cross-Foster_Experiment
# helpers increase survival of offspring in their first year and subsequent adult survival
# fledgling (F), old fledgling (O), subordinate (S) and primary (P)
# After the first year of life old fledgling become either a subordinate or primary
# The transition probabilities from fledgling to old fledgling, and from old fledgling to subordinate were
# no evidence that survival differed between subordinates and primaries
# paper contains estimates of probabilities for transitioning between stages

# https://www.sciencedirect.com/science/chapter/bookseries/abs/pii/S0065345407370046
# The fact that both dominant and subordinate females may lay eggs within the nest adds complexity to the issue of sex allocation that needs to be explored. 

# https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/1365-2656.12849
# how does food availability, group composition, age, sex, population density promote subordinate between-group dispersal 
# individuals that joined another group as non-natal subordinates were mainly female
# between-group dispersing females often became cobreeders, obtaining maternity in the new territory, and were likely to inherit the territory in the future, leading to higher lifetime reproductive success compared to females that floated. 
# results suggest subordinate between-group dispersal provides reproductive benefits when dispersing to independent breeding position is limited
# there is a cost to being floaters due to increased predation and reduced access to food
# evicted indiviudals may join other groups to avoid becoming floaters 

# https://animalecologyinfocus.com/2018/08/21/why-island-birds-sometimes-move-in-with-strangers/
# Why island birds sometimes move in with strangers

# https://pure.rug.nl/ws/portalfiles/portal/6727689/2009BehavEcolEikenaar.pdf
# Males were older than females when acquiring a territory for the first time. 
# for both sexes, the proximity of natal territory to a vacant dominant position was positively related to the individual’s chance of claiming the vacancy
# Older males were more likely to gain vacant dominant positions, age did not affect territory acquisition in females
# territory ownership is a prerequisite for male reproduction, whereas females can reproduce on their natal territory

# https://link.springer.com/article/10.1007/BF00167742
# two or more alloparents in medium-quality territories significantly decreased reproductive success
# this may have been due to the joint-nesting and reproductive competition that could occur in breeding groups
# or simply to resource depression when a large number of previous offspring remained on their natal territory

# https://pmc.ncbi.nlm.nih.gov/articles/PMC4718175/

# https://pmc.ncbi.nlm.nih.gov/articles/PMC6405499/



#__________________________________
# CLASSES
#__________________________________

# MALE
# fledgling: become subordinate (automatic)
# subordinate: disperse
# primary: reproduce (automatic), evict subordinate, adjust teritory, remove eggs
# floater: become subordinate, become primary, claim teritory

# FEMALE
# fledgling: become subordinate (automatic)
# subordinate: disperse
# primary: reproduce (automatic, weight sex of offspring), evict subordinate, remove eggs
# floater: become subordinate, become primary

#__________________________________
# ACTIONS
#__________________________________

# reproduce: produce 1-3 fledglings (weighted towards 1). Fitness of fledglings determined by number of subordinates and teritory size
# disperse: becoming floater, subordinate, or primary
# become primary: competition for primary position after primary death
# claim teritory: claim teritory greater than minimum size 
# adjust teritory: minor adjustment in teritory center
# evict subordinate: remove subordinate 
# remove eggs: prevent subordinate atempt at producing offspring
# become subordinate: floater or subordinate can become subordinate, this occurs automatically for offspring in thier natal teritory

#__________________________________
# FITNESS AND SURVIVAL
#__________________________________

# fitness fledgling: teritory size, helpers
# fitness global: age, class (fledgling, subordinate, primary, floater)
# fitness individual (sub, prim, float): upbringing (teritory size, helpers), cost of reproduction (reduced by assistance), current teritory size (density dependent with sub, prim, and fledg) 
# survival: fitness dependant

#__________________________________
# SIMULATION PARAMETERS
#__________________________________

# teritory diameter <- male age, numper of helpers
# teritory adjust distance 
# grid value -> carrying capacity -> density depenent selection
# dispersal distance
# age based fitness -> fitness
# class based fitness -> fitness
# reproductive cost -> fitness
# impact of teritory size -> fledgling fitness & adult fitness
# impact of subordinates -> fledgling fitness & adult fitness & reduction in reproductive cost from subordinates

#__________________________________
# SIMULATION METRICS
#__________________________________

# survival by sex, age, and class
# number of subordinates
# population size (if too high, simulation ends)
# action choice 





