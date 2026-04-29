import random

import numpy as np


class TerritoryMap:
    def __init__(self, pop, habitat_quality, diameter, min_quality):
        self.habitat_quality = habitat_quality
        self.pop = pop
        self.dims = habitat_quality.shape
        self.territory_map = None
        self.diameter = diameter
        self.min_quality = min_quality
        self.territory_dict = {}
        self.current_year = 0

    def set_year(self, year):
        self.current_year = year

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
            distances = np.linalg.norm(np.array(centers) - center, axis=1)  # euclidian distances
            territory = max(self.territory_dict.keys()) + 1
        else:
            # first territory
            distances = np.array((self.diameter))
            territory = 1

        if np.min(distances) >= self.diameter / 4:
            # check territory size
            new_territory = {"territory": territory, "center": center}

            if self.update(new_territory)["quality"] >= self.min_quality: # type: ignore
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
                    "distance_map": None,
                }

                self.pop.update_life_history(primary_male, "primary", new_territory["territory"])

                # territory creation was succeful
                success = True

                self.update()

        return success

    def update(self, new_territory=None):
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

        if new_territory is not None:
            territories.append(new_territory["territory"])
            centers.append(new_territory["center"])
            distance_maps.append(None)

        # create array
        proximity = np.tile(0, (self.dims[0], self.dims[1], len(territories) + 1))

        coordinates = np.stack(np.indices(self.dims), axis=-1).reshape(-1, 2)

        for center, i, distance_map, territory in zip(centers, range(len(centers)), distance_maps, territories):
            if distance_map is None:
                new_distance_map = (self.diameter / 2) - np.linalg.norm(coordinates - center, axis=1).reshape(self.dims)
                proximity[:, :, i + 1] = new_distance_map
                if new_territory is not None:
                    if territory != new_territory["territory"]:
                        self.territory_dict[territory]["distance_map"] = new_distance_map.copy()
            else:
                proximity[:, :, i + 1] = self.territory_dict[territory]["distance_map"].copy()

        # get maxima for each x,y value across z axis
        max_proximity = proximity.max(axis=2, keepdims=True)

        # get mask for all x,y values equal to the maxima that are not equal to 0
        max_mask = np.logical_and(proximity == max_proximity, np.repeat((max_proximity != 0), proximity.shape[2], axis=2))

        # select teritory claims by randomly assigning numbers to all coordinates equal to the x,y maxima
        territory_map = np.argmax(np.where(max_mask, np.random.random(max_mask.shape), -1), axis=2)

        # convert to territory
        territories = np.array([0] + territories)
        territory_map = territories[territory_map]

        if new_territory is not None:
            return {
                "size": np.sum(territory_map == new_territory["territory"]),
                "quality": np.sum((territory_map == new_territory["territory"]) * self.habitat_quality),
                "territory_map": territory_map,
            }

        self.territory_map = territory_map

        for territory in self.territory_dict.keys():
            self.territory_dict[territory]["size"] = np.sum(self.territory_map == territory)
            self.territory_dict[territory]["quality"] = np.sum((self.territory_map == territory) * self.habitat_quality)
            self.territory_map = territory_map

        self.check_territories()  # remove territories without the required habitat quality

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
        else:  # female
            self.territory_dict[territory]["primary_female_competition"].append(ind)

    def decide_primary(self, territory, sex):
        new_primary = None

        if sex == "male":
            if len(self.territory_dict[territory]["primary_male_competition"]) > 0:
                # age based competition, competing male with max age wins
                ages = []
                for ind in self.territory_dict[territory]["primary_male_competition"]:
                    ages.append(self.current_year - self.pop[ind]["year"])

                new_primary = self.territory_dict[territory]["primary_male_competition"][np.argmax(ages)]

        else:  # female
            if len(self.territory_dict[territory]["primary_female_competition"]) > 0:
                # the winning female is chosen uniformly randomly between competing females
                new_primary = random.choice(self.territory_dict[territory]["primary_female_competition"])

        # if a new primary is chosen
        if new_primary is not None:
            # update life history and territory of individual
            self.pop.update_life_history(new_primary, "primary", territory)

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
