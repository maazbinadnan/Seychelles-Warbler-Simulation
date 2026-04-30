import numpy as np

from kinship import Kinship
from population import Population
from territory import TerritoryMap


class GeneticController:
    def __init__(self, pop: Population, territory_map: TerritoryMap, kinship: Kinship, start_year: int, min_kinship: float) -> None:
        self.pop = pop
        self.territory_map = territory_map
        self.kinship = kinship
        self.start_year = start_year
        self.min_kinship = min_kinship
        self.year = start_year
        self.genome_keys = ("w_kin", "w_qual", "w_risk")

    def _set_year(self, year):
        self.year = year

    def _get_genome(self, ind):
        genome = self.pop[ind].get("genome")
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

        if life_history == "floater":
            return self._decide_floater(ind)

        if life_history == "fledgling":
            return self._decide_fledgling(ind)

        if life_history == "subordinate":
            return self._decide_subordinate(ind)

        return self.pop[ind]["territory"], (-1, -1), "nothing"

    def _find_primary_vacancy(self, ind, sex: str, life_history: str):
        primary_key = "primary_female" if sex == "female" else "primary_male"
        competition_key = "primary_female_competition" if sex == "female" else "primary_male_competition"
        territories = self.territory_map.get_territories()

        if life_history == "subordinate":
            own_territory = self.pop[ind]["territory"]
            if own_territory in territories and territories[own_territory][primary_key] is None:
                return own_territory
            return None

        candidates = [
            (territory_id, len(info[competition_key]))
            for territory_id, info in territories.items()
            if info[primary_key] is None
        ]
        if not candidates:
            return None
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

    def _decide_floater(self, ind):
        territory = self.pop[ind]["territory"]
        sex = self.pop[ind]["sex"]
        genome = self._get_genome(ind)

        if self.year == self.start_year:
            if sex == "male":
                center = (np.random.randint(0, self.territory_map.dims[0]), np.random.randint(0, self.territory_map.dims[1]))
                return territory, center, "establish_territory"
            return territory, (-1, -1), "nothing"

        if sex == "female":
            territory_id = self._find_primary_vacancy(ind, sex="female", life_history="floater")
            if territory_id is not None:
                return territory_id, (-1, -1), "compete_primary"

            best_territory = self._find_best_subordinate_territory(ind)
            if best_territory is not None:
                return best_territory, (-1, -1), "request_subordinate"

            territories = self.territory_map.get_territories()
            if territories and genome["w_risk"] > 0.5:
                random_territory = np.random.choice(list(territories.keys()))
                return random_territory, (-1, -1), "request_subordinate"

            return territory, (-1, -1), "nothing"

        if sex == "male":
            territory_id = self._find_primary_vacancy(ind, sex="male", life_history="floater")
            if territory_id is not None:
                return territory_id, (-1, -1), "compete_primary"

            center = (np.random.randint(0, self.territory_map.dims[0]), np.random.randint(0, self.territory_map.dims[1]))
            return territory, center, "establish_territory"

        return territory, (-1, -1), "nothing"

    def _decide_fledgling(self, ind):
        territory = self.pop[ind]["territory"]
        sex = self.pop[ind]["sex"]
        genome = self._get_genome(ind)

        if sex == "female" and territory in self.territory_map.get_territories():
            quality = self._territory_quality(territory)
            primary_female = self.territory_map[territory].get("primary_female")
            relatedness = self._relatedness(ind, primary_female)
            stay_score = genome["w_kin"] * relatedness + genome["w_qual"] * quality
            disperse_score = genome["w_risk"] * (1.0 - relatedness)
            if stay_score >= disperse_score:
                return territory, (-1, -1), "request_subordinate"
            return territory, (-1, -1), "disperse"

        if sex == "male":
            disperse_score = genome["w_risk"]
            stay_score = genome["w_kin"] * self.min_kinship
            if stay_score >= disperse_score:
                return territory, (-1, -1), "request_subordinate"
            return territory, (-1, -1), "disperse"

        return territory, (-1, -1), "nothing"

    def _decide_subordinate(self, ind):
        sex = self.pop[ind]["sex"]
        territory = self.pop[ind]["territory"]
        genome = self._get_genome(ind)
        territories = self.territory_map.get_territories()

        if territory not in territories:
            return territory, (-1, -1), "nothing"

        if sex == "female":
            to_compete_territory = self._find_primary_vacancy(ind, sex="female", life_history="subordinate")
            if to_compete_territory is not None:
                return to_compete_territory, (-1, -1), "compete_primary"

            primary_female = territories[territory].get("primary_female")
            relatedness = self._relatedness(ind, primary_female)
            stay_score = genome["w_kin"] * relatedness + genome["w_qual"] * self._territory_quality(territory)
            disperse_score = genome["w_risk"] * (1.0 - relatedness)
            if stay_score >= disperse_score:
                return territory, (-1, -1), "nothing"
            return territory, (-1, -1), "disperse"

        if sex == "male":
            to_compete_territory = self._find_primary_vacancy(ind, sex="male", life_history="subordinate")
            if to_compete_territory is not None:
                return to_compete_territory, (-1, -1), "compete_primary"

            primary_male = territories[territory].get("primary_male")
            relatedness = self._relatedness(ind, primary_male)
            if genome["w_kin"] * relatedness >= genome["w_risk"]:
                return territory, (-1, -1), "nothing"
            return territory, (-1, -1), "disperse"

        return territory, (-1, -1), "nothing"

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