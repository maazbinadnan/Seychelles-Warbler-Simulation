from population import Population
from territory import TerritoryMap
from kinship import Kinship
import numpy as np
from typing import Tuple
from typing import Any

class ruleBasedAI():
    def __init__(self, pop : Population, territory_map : TerritoryMap, kinship: Kinship, start_year: int, min_kinship: float) -> None:
        self.pop = pop
        self.territory_map = territory_map
        self.kinship = kinship
        self.start_year = start_year
        self.min_kinship = min_kinship

    def _set_year(self, year):
        self.year = year


    def action(self, ind) -> Tuple[int | None, tuple[int, int], str]:
        life_history = self.pop[ind]["life_history"]

        if life_history == "floater":
            return self._decide_floater(ind)
        
        if life_history == "fledgling":
            return self._decide_fledgling(ind)
        
        if life_history == "subordinate":
            return self._decide_subordinate(ind)
        else:
            return self.pop[ind]["territory"], (-1,-1), "nothing"

    '''
    HELPER FUNCTIONS
    '''
    def _find_primary_vacancy(self, ind, sex: str, life_history: str) -> int | None:
        
        '''Find a territory with a vacant primary slot for the given sex.
        Subordinate: only consider their own territory.
        Floater: scan all territories, pick the one with the least competition.'''

        primary_key = "primary_female" if sex == "female" else "primary_male"
        competition_key = "primary_female_competition" if sex == "female" else "primary_male_competition"
        territories = self.territory_map.get_territories()

        if life_history == "subordinate":
            own_territory = self.pop[ind]["territory"]
            if own_territory in territories and territories[own_territory][primary_key] is None:
                return own_territory
            return None

        # Floater: find all vacancies and pick the one with the least competition
        candidates = [
            (territory_id, len(info[competition_key]))
            for territory_id, info in territories.items()
            if info[primary_key] is None
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda x: x[1])[0]

    def _is_high_quality_territory(self, territory_id: int) -> bool:
        '''Heuristic: high quality > 3.5, low quality <= 3.5'''
        territories = self.territory_map.get_territories()
        if territory_id not in territories:
            return False
        quality = territories[territory_id].get("quality")
        if quality is None:
            return False
        return float(quality) >= 3.5

    def _find_best_subordinate_territory(self, ind) -> int | None:
        '''Ranks territories for a female floater requesting subordinate status using 5 heuristics:
        1. Territory must be high-quality (>= threshold).
        2. Territory must have capacity: quality > current group size (2 + subordinates).
        3. Higher relatedness to primary_female ranked first.
        4. Fewer existing subordinates preferred (lower group density).
        5. Fewer pending subordinate requests preferred (better programmatic odds).
        Returns the best territory_id, or None if no suitable territory exists.'''
        territories = self.territory_map.get_territories()
        scored = []

        for territory_id, info in territories.items():
            quality = info.get("quality")
            if quality is None:
                continue

            # Heuristic 1: must be high quality
            if not self._is_high_quality_territory(territory_id):
                continue

            # Heuristic 2: must have capacity
            group_size = 2 + len(info["subordinates"])
            if quality <= group_size:
                continue

            # Heuristic 3: relatedness to primary female
            primary_female = info.get("primary_female")
            if primary_female is None:
                relatedness = 0.0
            else:
                try:
                    relatedness = float(self.kinship.calculate_relatedness(ind, primary_female)) # type: ignore
                except Exception:
                    relatedness = 0.0

            # Heuristics 4 & 5: lower is better, negate for descending sort
            num_subordinates = len(info["subordinates"])
            num_requests = len(info["subordinate_request"])

            score = (relatedness, -num_subordinates, -num_requests)
            scored.append((score, territory_id))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    '''
    MAIN DECISION FUNCTIONS
    '''
    def _decide_floater(self, ind) -> Tuple[int | None, tuple[int, int], str]:

        territory = self.pop[ind]["territory"]
        sex = self.pop[ind]["sex"]

        # Year 1 logic only
        if self.year == self.start_year:
            match sex:
                case "male":
                    center = (np.random.randint(0,self.territory_map.dims[0]), np.random.randint(0,self.territory_map.dims[1]))
                    return territory, center, "establish_territory"
                case "female":
                    return territory, (-1,-1), "nothing"
                case _:
                    return territory, (-1,-1), "nothing"
        else:
            if sex == "female":
                territory_id = self._find_primary_vacancy(ind, sex="female", life_history="floater")
                if territory_id is not None:
                    return territory_id, (-1,-1), "compete_primary"
                else:
                # fallback: request subordinate at a chosen territory with heuristics
                    best_territory = self._find_best_subordinate_territory(ind)
                if best_territory is not None:
                    return best_territory, (-1,-1), "request_subordinate"
                else:
                # last resort: random territory
                    territories = self.territory_map.get_territories()
                if territories:
                    random_territory = np.random.choice(list(territories.keys()))
                    return random_territory, (-1,-1), "request_subordinate"

            elif sex == "male":
                territory_id = self._find_primary_vacancy(ind, sex="male", life_history="floater")
                if territory_id is not None:
                    return territory_id, (-1,-1), "compete_primary"
                # fallback: establish territory at random coordinates
                center = (np.random.randint(0, self.territory_map.dims[0]), np.random.randint(0, self.territory_map.dims[1]))
                return territory, center, "establish_territory"
            return territory, (-1,-1), "nothing"


    def _decide_fledgling(self, ind) -> Tuple[int | None, tuple[int, int], str]:
        territory = self.pop[ind]["territory"]
        sex = self.pop[ind]["sex"]

        # Female fledglings: high-quality natal territory => help, else disperse.
        if sex == "female" and territory in self.territory_map.get_territories():
            if self._is_high_quality_territory(territory):
                return territory, (-1, -1), "request_subordinate"
            return territory, (-1, -1), "disperse"
        elif sex == "male": 
            return territory, (-1, -1), "disperse"
        return territory, (-1, -1), "nothing"


    def _decide_subordinate(self, ind) -> Tuple[int | None, tuple[int, int], str]:
        sex = self.pop[ind]["sex"]
        territory = self.pop[ind]["territory"]
        territories = self.territory_map.get_territories()

        if territory not in territories:
            return territory, (-1, -1), "nothing"

        if sex == "female":
            to_compete_territory = self._find_primary_vacancy(ind, sex="female", life_history="subordinate")
            if to_compete_territory is not None:
                return to_compete_territory, (-1,-1) , "compete_primary"
            
            # Check relatedness to the territory's primary female
            primary_female = territories[territory].get("primary_female")
            if primary_female is None:
                relatedness = 0.0
            else:
                relatedness = float(self.kinship.calculate_relatedness(ind, primary_female)) # type: ignore
            # Stay if sufficiently related, otherwise disperse
            if relatedness >= self.min_kinship:
                return territory, (-1, -1), "nothing"
            else:
                return territory, (-1, -1), "disperse"

        if sex == "male":
            to_compete_territory = self._find_primary_vacancy(ind, sex="male", life_history="subordinate")
            if to_compete_territory is not None:
                return to_compete_territory, (-1, -1), "compete_primary"
            else:
                return territory, (-1, -1), "disperse"

        # default: no action
        return territory, (-1, -1), "nothing"

    def evict_subordinate_male_primary(self, ind) -> int | None:
        # get the primary bird for that individual's territory
        territory = self.pop[ind]["territory"]
        territories = self.territory_map.get_territories()
        if territory not in territories:
            return False

        territory_info = territories[territory]
        primary_male = territory_info.get("primary_male")
        primary_female = territory_info.get("primary_female")

        if ind not in territory_info.get("subordinates", []):
            return False

        relatedness_to_male = 0.0
        if primary_male is not None:
            try:
                relatedness_to_male = float(self.kinship.calculate_relatedness(ind, primary_male)) # type: ignore
            except Exception:
                relatedness_to_male = 0.0

        relatedness_to_female = 0.0
        if primary_female is not None:
            try:
                relatedness_to_female = float(self.kinship.calculate_relatedness(ind, primary_female)) # type: ignore
            except Exception:
                relatedness_to_female = 0.0

        if relatedness_to_male < self.min_kinship or relatedness_to_female < self.min_kinship:
            return True

        subordinates = territory_info.get("subordinates", [])
        quality = territory_info.get("quality")
        if quality is None:
            return False

        current_group_size = 2 + len(subordinates)
        if current_group_size <= float(quality):
            return False

        excess = int(np.ceil(current_group_size - float(quality)))
        male_subordinates = []
        female_subordinates = []

        for subordinate in subordinates:
            if subordinate == ind:
                continue
            sex = self.pop[subordinate]["sex"]
            if sex == "male":
                male_subordinates.append(subordinate)
            else:
                female_subordinates.append((subordinate, self.kinship.calculate_relatedness(subordinate, primary_female) if primary_female is not None else 0.0))

        if self.pop[ind]["sex"] == "male":
            if excess > 0:
                return True
            return False

        if male_subordinates:
            return False

        if excess > 0:
            female_subordinates.sort(key=lambda item: item[1])
            return any(subordinate == ind for subordinate, _ in female_subordinates[:excess])

        return False

    def acccept_subordinate(self, ind) -> bool:
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

        try:
            relatedness_to_male = float(self.kinship.calculate_relatedness(ind, primary_male)) # type: ignore
        except Exception:
            relatedness_to_male = 0.0

        try:
            relatedness_to_female = float(self.kinship.calculate_relatedness(ind, primary_female)) # type: ignore
        except Exception:
            relatedness_to_female = 0.0

        if relatedness_to_male <= self.min_kinship or relatedness_to_female <= self.min_kinship:
            return False

        vacancy_count = int(float(quality) - current_group_size)
        if vacancy_count <= 0:
            return False

        if vacancy_count == 1 and self.pop[ind]["sex"] == "male":
            female_requests = [request for request in territory_info.get("subordinate_request", []) if self.pop[request]["sex"] == "female"]
            if female_requests:
                return False

        return True

    def acccept_subordinate_reproduction(self, ind) -> bool:
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

        # Primary male always accepts subordinate female co-breeding.
        male_accepts = True

        try:
            relatedness_to_female = float(self.kinship.calculate_relatedness(ind, primary_female)) # type: ignore
        except Exception:
            relatedness_to_female = 0.0

        female_accepts = relatedness_to_female > self.min_kinship

        return male_accepts and female_accepts
    




















































