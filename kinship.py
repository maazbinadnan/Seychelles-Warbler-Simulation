import numpy as np
import pandas as pd


class Kinship:
    def __init__(self, pop, min_kinship):
        inds = pop.get_inds()
        pop_size = pop.pop_size()
        self.pop = pop
        initial = np.tile(0.05, (pop_size, pop_size))
        np.fill_diagonal(initial, 1.0)
        self.matrix = pd.DataFrame(initial, columns=(inds), index=(inds))
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
        new_matrix = pd.DataFrame(np.tile(0.05, (size, size)), columns=(all_inds), index=(all_inds))

        # add values for old_inds
        new_matrix.loc[old_inds, old_inds] = self.matrix.values

        # calculate kinship of new cohort
        for ind in new_inds:
            mother = self.pop[ind]["mother"]
            father = self.pop[ind]["father"]

            # kinship
            new_matrix.loc[:, ind] = np.multiply(0.5, np.add(new_matrix[mother].values, new_matrix[father].values))  # column
            new_matrix.loc[ind] = new_matrix.loc[:, ind]  # row
            new_matrix.loc[ind, ind] = 1.0

        # update matrix
        self.matrix = new_matrix.round(4)

    def return_df(self):
        return self.matrix

    def calculate_relatedness(self, ind1, ind2):
        return self.matrix.loc[ind1, ind2]

    def remove_outdated(self):
        inds = self.matrix.columns
        current_inds = self.pop.get_inds()
        common_inds = list(set(inds) & set(current_inds))

        # list all parents
        parents = set()
        for ind in self.pop.get_inds():
            parents.add(self.pop[ind]["mother"])
            parents.add(self.pop[ind]["father"])

        # for each individual in the matrix
        for ind in inds:
            # if indiviudal is not currently in the population
            if ind not in current_inds:
                # if individual is not the mother or father of any living indiviual
                if ind not in parents:
                    # get kinship values for individual
                    ind_kinship = self.matrix[ind][self.matrix[ind] < 1.0]
                    ind_kinship = ind_kinship.loc[common_inds]

                    # if it has a maximum kinship of less than the minimum to living individuals
                    if len(ind_kinship) == 0 or max(ind_kinship) < self.min_kinship:

                        # remove the row and column of the individual
                        self.matrix.drop(columns=ind, inplace=True)
                        self.matrix.drop(index=ind, inplace=True)
