import numpy as np

class utilityBasedAI():
    def __init__(self, pop, territory_map, kinship, min_kinship, year, diameter, min_quality):
        self.pop = pop
        self.territory_map = territory_map
        self.kinship = kinship
        self.min_kinship = min_kinship
        self.year = year
        self.diameter = diameter
        self.min_quality = min_quality
    
    def _get_age(self, ind):
        if "age" in self.pop[ind]:
            return self.pop[ind]["age"]
        return self.year - self.pop[ind]["year"]
    
    def _random_center(self):
        return (
            np.random.randint(0, self.territory_map.habitat_quality.shape[0]),
            np.random.randint(0, self.territory_map.habitat_quality.shape[1])
        )
    
    def _local_quality(self, center, radius):
        x, y = center
        x_min = max(0, x - radius)
        x_max = min(self.territory_map.habitat_quality.shape[0], x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(self.territory_map.habitat_quality.shape[1], y + radius + 1)

        patch = self.territory_map.habitat_quality[x_min:x_max, y_min:y_max]
        if patch.size == 0:
            return 0.0
        return float(np.sum(patch))
    
    def _is_high_quality(self, quality):
        return quality >= (self.min_quality * 2.0)
    
    def _relatedness_to_primary_female(self, ind, territory):
        territories = self.territory_map.get_territories()
        primary_female = territories[territory]["primary_female"]

        if primary_female is None:
            return 0.0

        try:
            return float(self.kinship.calculate_relatedness(ind, primary_female))
        except Exception:
            return float(self.kinship.matrix.loc[ind, primary_female])
        
    def set_year(self, year):
        self.year = year
        
    
    def _territory_quality(self, territory_id):
        territories = self.territory_map.get_territories()
        if territory_id not in territories:
            return 0.0

        quality = territories[territory_id]["quality"]
        if quality is not None:
            return quality

        center = territories[territory_id]["center"]
        return self._local_quality(center, radius=max(1, self.diameter // 4))
    
    def _decide_fledgling(self, ind, natal_territory, fallback_center):
        if natal_territory not in self.territory_map.get_territories():
            return "disperse", None, fallback_center

        utility_stay = (self._relatedness_to_primary_female(ind, natal_territory) * self._territory_quality(natal_territory)) + (self._territory_quality(natal_territory) * 0.005)

        vacant = 0
        total_vacancy_quality = 0
        for t_id, t_info in self.territory_map.get_territories().items():
            if t_info["primary_male"] is None or t_info["primary_female"] is None:
                vacant += 1
                total_vacancy_quality += t_info["quality"] if t_info["quality"] is not None else 0.0

        vacancy_rate = vacant / max(1, len(self.territory_map.get_territories()))
        avg_vacancy_quality = total_vacancy_quality / max(1, vacant)
        utility_disperse = 0.01 + (vacancy_rate * avg_vacancy_quality * 0.005)

        if utility_stay > utility_disperse:
            return "request_subordinate", natal_territory, fallback_center
        return "disperse", None, fallback_center

    
    def _decide_subordinate(self,ind,territory,sex,age,fallback_center):
        
        territories = self.territory_map.get_territories()
        
        if territory not in territories:
            return "disperse", None, fallback_center

        
        return "compete_primary", territory, fallback_center

        
        
    def _decide_floater(self, ind, territory, sex, age, fallback_center):
        compete_primary_utility = 0
        vacancy_key = "primary_male" if sex == "male" else "primary_female"
        best_vacant_territory = None
        best_vacant_quality = -1
        kinship_quality_ratio = []

        # --- Loop 1: find vacant territories for compete_primary ---
        vacant_ids = []
        vacant_qualities = []
        for t_id, t_info in self.territory_map.get_territories().items():
            if t_info[vacancy_key] is None:
                q = t_info["quality"] if t_info["quality"] is not None else 0.0
                vacant_ids.append(t_id)
                vacant_qualities.append(q)

        if len(vacant_ids) > 0:
            weights = np.array(vacant_qualities, dtype=float)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(vacant_ids)) / len(vacant_ids)
            best_vacant_territory = np.random.choice(vacant_ids, p=weights)
            best_vacant_quality = self._territory_quality(best_vacant_territory)

        # --- Loop 2: calculate kinship-quality ratios for request_subordinate ---
        for t_id, t_info in self.territory_map.get_territories().items():
            request_quality = t_info["quality"] if t_info["quality"] is not None else 0.0
            kinship = self._relatedness_to_primary_female(ind, t_id)
            kinship_quality_ratio.append((request_quality * kinship, t_id))

        if len(kinship_quality_ratio) > 0:
            best_kq_ratio, territory_request_subordinate = max(kinship_quality_ratio)
        else:
            best_kq_ratio = -1
            territory_request_subordinate = None

        if best_vacant_territory is not None:
            compete_primary_utility = best_vacant_quality * 0.5 * 0.5
        else:
            compete_primary_utility = -1

        request_subordinate_utility = best_kq_ratio

        # --- Establish territory (males only) ---
        if sex == "male":
            best_establish_quality = -1
            best_establish_center = None
            for i in range(3):
                candidate = (
                    np.random.randint(0, self.territory_map.habitat_quality.shape[0]),
                    np.random.randint(0, self.territory_map.habitat_quality.shape[1])
                )
                result = self.territory_map.update({"territory": 9999, "center": candidate})
                if result["quality"] > best_establish_quality:
                    best_establish_quality = result["quality"]
                    best_establish_center = candidate
            establish_territory_utility = best_establish_quality * 0.5
        else:
            establish_territory_utility = -1
            best_establish_center = None

        nothing_utility = 0.01

        best_utility = max(
            compete_primary_utility,
            request_subordinate_utility,
            establish_territory_utility,
            nothing_utility
        )

        if best_utility == compete_primary_utility and best_vacant_territory is not None:
            return "compete_primary", best_vacant_territory, fallback_center
        elif best_utility == establish_territory_utility and sex == "male":
            return "establish_territory", None, best_establish_center
        elif best_utility == request_subordinate_utility and territory_request_subordinate is not None:
            return "request_subordinate", territory_request_subordinate, fallback_center
        else:
            return "nothing", None, fallback_center



    def decide(self, ind):
        life_history = self.pop[ind]["life_history"]
        sex = self.pop[ind]["sex"]
        territory = self.pop[ind]["territory"]
        age = self._get_age(ind)

        center = self._random_center()

        if life_history == "fledgling":
            return self._decide_fledgling(ind, territory, center)

        if life_history == "subordinate":
            return self._decide_subordinate(ind, territory, sex, age, center)

        if life_history == "floater":
            return self._decide_floater(ind, territory, sex, age, center)

        # primaries stay in place during action phase
        return "nothing", territory, center
    
    def decide_evict_subordinate(self, primary_male, primary_female, subordinate, territory):
        """
        Returns (male_evicts: bool, female_evicts: bool).

        Primary female: evicts if the subordinate is unrelated (kinship below
        min_kinship to herself) — no inclusive fitness benefit in tolerating them.

        Primary male: evicts if territory quality is low and already has more
        than one subordinate — resource pressure outweighs any helper benefit.
        """
        quality = self._territory_quality(territory)
        territories = self.territory_map.get_territories()
        num_subordinates = len(territories[territory]["subordinates"])

        # Primary female decision: kin-based tolerance
        relatedness = self._relatedness_to_primary_female(subordinate, territory)
        female_evicts = relatedness < self.min_kinship

        # Primary male decision: resource constraint when territory is poor
        male_evicts = (not self._is_high_quality(quality)) and (num_subordinates > 1)

        return male_evicts, female_evicts

    def decide_accept_subordinate(self, primary_male, primary_female, candidate, territory):
        """
        Returns (male_accepts: bool, female_accepts: bool).

        Both primaries must agree. Acceptance is driven by kin selection
        (related candidate → shared fitness) and habitat quality (high-quality
        territory can support an extra helper).
        """
        quality = self._territory_quality(territory)
        relatedness = self._relatedness_to_primary_female(candidate, territory)

        is_related = relatedness >= self.min_kinship
        is_rich = self._is_high_quality(quality)

        # Primary female: accept related individuals, or related ones on any territory
        female_accepts = is_related

        # Primary male: accept if territory is high quality OR candidate is related
        male_accepts = is_rich or is_related

        return male_accepts, female_accepts

    def decide_subordinate_reproduction(self, primary_male, primary_female, subordinate_female, territory):
        """
        Returns (male_allows: bool, female_allows: bool).

        Subordinate reproduction is only permitted on high-quality (surplus)
        territories — the ecological constraints / benefit-of-philopatry framework.
        Primary female additionally suppresses unrelated subordinate females
        (reproductive competition); she tolerates related subordinates (daughters).
        """
        quality = self._territory_quality(territory)
        relatedness = self._relatedness_to_primary_female(subordinate_female, territory)

        is_rich = self._is_high_quality(quality)
        is_related = relatedness >= self.min_kinship

        # Primary female: allow only related subordinates (kin), and only on rich territories
        female_allows = is_rich and is_related

        # Primary male: allow on rich territories regardless of relatedness
        male_allows = is_rich

        return male_allows, female_allows

    def _set_year(self, year):
        self.year = year

    def action(self, ind):
        action, territory, center = self.decide(ind)
        return territory, center, action

    def evict_subordinate_male_primary(self, ind):
        territory = self.pop[ind]["territory"]
        territories = self.territory_map.get_territories()
        if territory not in territories:
            return False
        t_info = territories[territory]
        primary_male = t_info["primary_male"]
        primary_female = t_info["primary_female"]
        male_evicts, female_evicts = self.decide_evict_subordinate(
            primary_male, primary_female, ind, territory
        )
        return male_evicts or female_evicts

    def acccept_subordinate(self, ind):
        territory = self.pop[ind]["territory"]
        territories = self.territory_map.get_territories()
        if territory not in territories:
            return False
        t_info = territories[territory]
        primary_male = t_info["primary_male"]
        primary_female = t_info["primary_female"]
        male_accepts, female_accepts = self.decide_accept_subordinate(
            primary_male, primary_female, ind, territory
        )
        return male_accepts and female_accepts

    def acccept_subordinate_reproduction(self, ind):
        territory = self.pop[ind]["territory"]
        territories = self.territory_map.get_territories()
        if territory not in territories:
            return False
        t_info = territories[territory]
        primary_male = t_info["primary_male"]
        primary_female = t_info["primary_female"]
        male_allows, female_allows = self.decide_subordinate_reproduction(
            primary_male, primary_female, ind, territory
        )
        return male_allows and female_allows
