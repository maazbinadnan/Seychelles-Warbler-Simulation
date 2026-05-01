import numpy as np

from kinship import Kinship
from population import Population
from territory import TerritoryMap


class GeneticController:
    def __init__(self, pop: Population, territory_map: TerritoryMap, kinship: Kinship, start_year: int, min_kinship: float, establish_samples = 8,base_cost: float = 1.0) -> None:
        self.pop = pop
        self.territory_map = territory_map
        self.kinship = kinship
        self.start_year = start_year
        self.min_kinship = min_kinship
        self.year = start_year
        self.genome_keys = ("w_kin", "w_qual", "w_risk")
        self.establish_samples = establish_samples
        self.base_cost = base_cost

    def _set_year(self, year):
        self.year = year

    def _get_genome(self, ind):
        genome = self.pop[ind].get("genome")
        #fall back, where it sets it all to 0.5 random
        if genome is None:
            return {key: 0.5 for key in self.genome_keys}
        return {key: float(genome.get(key, 0.5)) for key in self.genome_keys}

    def _relatedness(self, ind1, ind2) -> float:
        if ind2 is None:
            return 0.0
        try:
            return float(self.kinship.calculate_relatedness(ind1, ind2))
        except Exception:
            return 0.0

    def _territory_quality(self, territory) -> float:
        territories = self.territory_map.get_territories()
        if territory not in territories:
            return 0.0
        quality = territories[territory].get("quality")
        if quality is None:
            return 0.0
        return float(quality)

    def action(self, ind):
        life_history = self.pop[ind]["life_history"]

        candidates = self._candidate_actions(ind, life_history)
        if not candidates:
            return self.pop[ind]["territory"], (-1, -1), "nothing"

        best_action, best_territory, best_center, _ = max(candidates, key=lambda item: item[3])
        return best_territory, best_center, best_action

    def _candidate_actions(self, ind, life_history):
        sex = self.pop[ind]["sex"]
        territory = self.pop[ind]["territory"]
        candidates = [("nothing", territory, (-1, -1), self._score_action(ind, "nothing", territory, None))]

        if life_history == "floater":
            if sex == "female":
                #first search for primary vacancies, then subordinate territories, then random territories, same logic as rulebased, but instead this is scored using genome
                primary_vacancy = self._find_primary_vacancy(ind, sex="female", life_history="floater")
                if primary_vacancy is not None:
                    candidates.append(("compete_primary", primary_vacancy, (-1, -1), self._score_action(ind, "compete_primary", primary_vacancy, None)))

                subordinate_territory = self._find_best_subordinate_territory(ind)
                if subordinate_territory is not None:
                    candidates.append(("request_subordinate", subordinate_territory, (-1, -1), self._score_action(ind, "request_subordinate", subordinate_territory, None)))

                territories = self.territory_map.get_territories()
                if territories:
                    random_territory = np.random.choice(list(territories.keys()))
                    candidates.append(("request_subordinate", random_territory, (-1, -1), self._score_action(ind, "request_subordinate", random_territory, None)))

            elif sex == "male":
                primary_vacancy = self._find_primary_vacancy(ind, sex="male", life_history="floater")
                if primary_vacancy is not None:
                    candidates.append(("compete_primary", primary_vacancy, (-1, -1), self._score_action(ind, "compete_primary", primary_vacancy, None)))

                center = self._best_establish_center(ind)
                candidates.append(("establish_territory", territory, center, self._score_action(ind, "establish_territory", territory, center)))

        elif life_history == "fledgling":
            if sex == "female":
                candidates.append(("request_subordinate", territory, (-1, -1), self._score_action(ind, "request_subordinate", territory, None)))
                candidates.append(("disperse", territory, (-1, -1), self._score_action(ind, "disperse", territory, None)))
            elif sex == "male":
                candidates.append(("request_subordinate", territory, (-1, -1), self._score_action(ind, "request_subordinate", territory, None)))
                candidates.append(("disperse", territory, (-1, -1), self._score_action(ind, "disperse", territory, None)))

        elif life_history == "subordinate":
            if sex == "female":
                primary_vacancy = self._find_primary_vacancy(ind, sex="female", life_history="subordinate")
                if primary_vacancy is not None:
                    candidates.append(("compete_primary", primary_vacancy, (-1, -1), self._score_action(ind, "compete_primary", primary_vacancy, None)))
                candidates.append(("disperse", territory, (-1, -1), self._score_action(ind, "disperse", territory, None)))
                candidates.append(("nothing", territory, (-1, -1), self._score_action(ind, "nothing", territory, None)))

            elif sex == "male":
                primary_vacancy = self._find_primary_vacancy(ind, sex="male", life_history="subordinate")
                if primary_vacancy is not None:
                    candidates.append(("compete_primary", primary_vacancy, (-1, -1), self._score_action(ind, "compete_primary", primary_vacancy, None)))
                candidates.append(("disperse", territory, (-1, -1), self._score_action(ind, "disperse", territory, None)))
                candidates.append(("nothing", territory, (-1, -1), self._score_action(ind, "nothing", territory, None)))

        return candidates

    def _score_action(self, ind, action, territory, center) -> float:
        genome = self._get_genome(ind)

        if action == "nothing":
            return 0.0

        if action == "disperse":
            return -1.0 + (0.25 * genome["w_risk"])

        if action == "request_subordinate":
            return self._score_request_subordinate(ind, territory, genome)

        if action == "compete_primary":
            return self._score_compete_primary(ind, territory, genome)

        if action == "establish_territory":
            return self._score_establish_territory(ind, center, genome)

        return -1.0

    def _score_request_subordinate(self, ind, territory, genome) -> float:
        territories = self.territory_map.get_territories()
        if territory not in territories:
            return -1.0

        territory_info = territories[territory]
        quality = float(territory_info.get("quality") or 0.0)
        subordinates = territory_info.get("subordinates", [])
        current_group_size = 2 + len(subordinates)

        primary_male = territory_info.get("primary_male")
        primary_female = territory_info.get("primary_female")
        relatedness_to_male = self._relatedness(ind, primary_male)
        relatedness_to_female = self._relatedness(ind, primary_female)
        kin_gain = (relatedness_to_male + relatedness_to_female) / 2.0
        #direct gain is the quality of the territory, divided by the current group size (including the primary pair), so that larger groups have diminishing returns, and smaller groups have higher returns
        direct_gain = quality / max(current_group_size, 1)
        cost = max(0.0, current_group_size - quality) + (1.0 - kin_gain)
        #finally manage it as an individual's cost + the direct fitness + indirect fitness
        return (genome["w_kin"] * kin_gain * quality) + (genome["w_qual"] * direct_gain) - (genome["w_risk"] * cost)

    def _score_compete_primary(self, ind, territory, genome) -> float:
        territories = self.territory_map.get_territories()
        if territory not in territories:
            return -1.0

        territory_info = territories[territory]
        quality = float(territory_info.get("quality") or 0.0)
        competition_key = "primary_female_competition" if self.pop[ind]["sex"] == "female" else "primary_male_competition"
        competition = len(territory_info.get(competition_key, []))

        primary_male = territory_info.get("primary_male")
        primary_female = territory_info.get("primary_female")
        subordinates = territory_info.get("subordinates", [])
        #check relatedness to all group members
        kin_targets = [target for target in (primary_male, primary_female, *subordinates) if target is not None]
        kin_gain = 0.0
        #check relatedness to each target
        if kin_targets:
            kin_gain = sum(self._relatedness(ind, target) for target in kin_targets) / len(kin_targets)

        #add a bonus if compete primary list is empty
        vacancy_bonus = 1.0 if competition == 0 else 0.0
        #hence, the direct gain would be quality of the territory, plus vacancy bonus
        direct_gain = quality + vacancy_bonus
        #cost is number of competitors, 
        cost = competition + (1.0 / max(quality, 1.0))
        
        #finally, the score is the weighted sum of kin gain (aka indirect fitness) and direct gain, minus the weighted cost
        return (genome["w_kin"] * kin_gain * quality) + (genome["w_qual"] * direct_gain) - (genome["w_risk"] * cost)


    def _best_establish_center(self, ind):
        #take a sample of 8 random territories and score them based on the individual's genome, then return the center thats scoired the best
        best_center = (np.random.randint(0, self.territory_map.dims[0]), np.random.randint(0, self.territory_map.dims[1]))
        best_score = -float("inf")
        for _ in range(self.establish_samples - 1):
            center = (np.random.randint(0, self.territory_map.dims[0]), np.random.randint(0, self.territory_map.dims[1]))
            score = self._score_establish_territory(ind, center, self._get_genome(ind))
            if score > best_score:
                best_center = center
                best_score = score

        return best_center

    def _score_establish_territory(self, ind, center, genome) -> float:
        if center is None:
            return 0.0

        test_territory = {
            "territory" : -1,
            "center" : center,
        }   
        #use territory map's update function
        test_quality = self.territory_map.update(new_territory=test_territory)
        quality = test_quality["quality"] if test_quality else 0.0
        
        # we define the base cost as the risk an individual carries + the base cost  + 1/their fitness -> lower fitness = higher cost
        cost = self.base_cost + (1.0/self.pop[ind]["fitness"])
        direct_gain = quality
        return (genome["w_qual"] * direct_gain) - (genome["w_risk"] * cost)

    def _find_primary_vacancy(self, ind, sex: str, life_history: str):
        primary_key = "primary_female" if sex == "female" else "primary_male"
        competition_key = "primary_female_competition" if sex == "female" else "primary_male_competition"
        territories = self.territory_map.get_territories()

        if life_history == "subordinate":
            own_territory = self.pop[ind]["territory"]
            if own_territory in territories and territories[own_territory][primary_key] is None:
                return own_territory
            return None

        #return all territories where primary is None, candiates stores [territory_id, number of competitors]
        candidates = [
            (territory_id, len(info[competition_key]))
            for territory_id, info in territories.items()
            if info[primary_key] is None
        ]
        if not candidates:
            return None
        #return the one with the least competition
        return min(candidates, key=lambda x: x[1])[0]

    def _is_high_quality_territory(self, territory_id: int) -> bool:
        return self._territory_quality(territory_id) >= 3.5

    def _find_best_subordinate_territory(self, ind):
        territories = self.territory_map.get_territories()
        genome = self._get_genome(ind)
        scored = []

        for territory_id, info in territories.items():
            quality = info.get("quality")
            if quality is None or not self._is_high_quality_territory(territory_id):
                continue

            group_size = 2 + len(info["subordinates"])
            if quality <= group_size:
                continue

            primary_female = info.get("primary_female")
            relatedness = self._relatedness(ind, primary_female)
            num_subordinates = len(info["subordinates"])
            num_requests = len(info["subordinate_request"])

            score = (
                genome["w_kin"] * relatedness
                + genome["w_qual"] * float(quality)
                - genome["w_risk"] * (num_subordinates + num_requests)
            )
            scored.append((score, territory_id))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def evict_subordinate_male_primary(self, ind):
        territory = self.pop[ind]["territory"]
        territories = self.territory_map.get_territories()
        if territory not in territories:
            return False

        territory_info = territories[territory]
        primary_male = territory_info.get("primary_male")
        primary_female = territory_info.get("primary_female")

        if ind not in territory_info.get("subordinates", []):
            return False

        genome = self._get_genome(ind)
        relatedness_to_male = self._relatedness(ind, primary_male)
        relatedness_to_female = self._relatedness(ind, primary_female)
        quality = territory_info.get("quality")
        if quality is None:
            return False

        current_group_size = 2 + len(territory_info.get("subordinates", []))
        pressure = max(0.0, current_group_size - float(quality))
        kin_pressure = genome["w_kin"] * min(relatedness_to_male, relatedness_to_female)
        risk_bias = genome["w_risk"] + pressure

        if kin_pressure < self.min_kinship or risk_bias > 0.8:
            return True

        subordinates = territory_info.get("subordinates", [])
        male_subordinates = [subordinate for subordinate in subordinates if subordinate != ind and self.pop[subordinate]["sex"] == "male"]
        if not male_subordinates and pressure > 0:
            return True

        return False

    def acccept_subordinate(self, ind):
        territory = self.pop[ind]["territory"]
        territories = self.territory_map.get_territories()
        if territory not in territories:
            return False

        territory_info = territories[territory]
        if ind not in territory_info.get("subordinate_request", []):
            return False

        primary_male = territory_info.get("primary_male")
        primary_female = territory_info.get("primary_female")
        if primary_male is None or primary_female is None:
            return False

        quality = territory_info.get("quality")
        if quality is None:
            return False

        current_group_size = 2 + len(territory_info.get("subordinates", []))
        if current_group_size >= float(quality):
            return False

        genome = self._get_genome(ind)
        relatedness_to_male = self._relatedness(ind, primary_male)
        relatedness_to_female = self._relatedness(ind, primary_female)
        score = genome["w_kin"] * min(relatedness_to_male, relatedness_to_female) + genome["w_qual"] * float(quality) - genome["w_risk"] * current_group_size

        return score >= self.min_kinship

    def acccept_subordinate_reproduction(self, ind):
        territory = self.pop[ind]["territory"]
        territories = self.territory_map.get_territories()
        if territory not in territories:
            return False

        territory_info = territories[territory]
        if ind not in territory_info.get("subordinate_request", []):
            return False

        if self.pop[ind]["sex"] != "female":
            return False

        primary_male = territory_info.get("primary_male")
        primary_female = territory_info.get("primary_female")
        if primary_male is None or primary_female is None:
            return False

        genome = self._get_genome(ind)
        relatedness_to_female = self._relatedness(ind, primary_female)
        quality = territory_info.get("quality")
        if quality is None:
            return False

        score = genome["w_kin"] * relatedness_to_female + genome["w_qual"] * float(quality) - genome["w_risk"]
        return score > self.min_kinship