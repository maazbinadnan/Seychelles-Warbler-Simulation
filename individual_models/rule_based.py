import random
random.seed(42)
import numpy as np


class ruleBasedAI():
    def __init__(self, pop, territory_map, kinship, min_kinship, year, diameter, min_quality):
        self.pop = pop
        self.territory_map = territory_map
        self.kinship = kinship
        self.min_kinship = min_kinship
        self.year = year
        self.diameter = diameter
        self.min_quality = min_quality

    def set_year(self, year):
        self.year = year

    def decide(self, ind):
        life_history = self.pop[ind]["life_history"]
        sex = self.pop[ind]["sex"]
        territory = self.pop[ind]["territory"]
        age = self._get_age(ind)
        
        #we define a fallback center just in case
        center = self._random_center()

        #we then define rules for each type of individual
        if life_history == "fledgling":
            return self._decide_fledgling(ind, sex, territory, center)

        if life_history == "subordinate":
            return self._decide_subordinate(ind, sex, age, territory, center)

        if life_history == "floater":
            return self._decide_floater(ind, sex, center)

        # primaries stay in place during action phase
        return "nothing", territory, center

    def _get_age(self, ind):
        if "age" in self.pop[ind]:
            return self.pop[ind]["age"]
        return self.year - self.pop[ind]["year"]

    def _random_center(self):
        return (
            np.random.randint(0, self.territory_map.habitat_quality.shape[0]),
            np.random.randint(0, self.territory_map.habitat_quality.shape[1])
        )

    def _territory_quality(self, territory_id):
        territories = self.territory_map.get_territories()
        
        #check just in case
        if territory_id not in territories:
            return 0.0

        quality = territories[territory_id]["quality"]
        if quality is not None:
            return quality

        center = territories[territory_id]["center"]
        return self._local_quality(center, radius=max(1, self.diameter // 4))

    def _local_quality(self, center, radius):
        x, y = center
        x_min = max(0, x - radius)
        x_max = min(self.territory_map.habitat_quality.shape[0], x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(self.territory_map.habitat_quality.shape[1], y + radius + 1)
        
        #takes the small patch of habitat quality around the center 
        patch = self.territory_map.habitat_quality[x_min:x_max, y_min:y_max]
        if patch.size == 0:
            return 0.0
        #sums it up to get the total quality of the territory
        return float(np.sum(patch))

    def _is_high_quality(self, quality):
        return quality >= (self.min_quality * 2.0)

    def _choose_highest_quality_territory(self, territory_ids):
        if len(territory_ids) == 0:
            return None

        qualities = [self._territory_quality(t) for t in territory_ids]
        return territory_ids[int(np.argmax(qualities))]

    def _relatedness_to_primary_female(self, ind, territory):
        territories = self.territory_map.get_territories()
        primary_female = territories[territory]["primary_female"]

        if primary_female is None:
            return 0.0

        try:
            return float(self.kinship.calculate_relatedness(ind, primary_female))
        except Exception:
            return float(self.kinship.matrix.loc[ind, primary_female])

    def _find_vacancy(self, sex):
        vacancy_key = "primary_male" if sex == "male" else "primary_female"
        vacancies = []

        for territory_id, info in self.territory_map.get_territories().items():
            #if no one is currently occupying the territory, it's a vacancy
            if info[vacancy_key] is None:
                #assign this territory as a potential vacancy for the individual
                vacancies.append(territory_id)

        return self._choose_highest_quality_territory(vacancies)

    def _integral_image(self, arr):
        """Compute 2-D summed-area table for fast rectangular patch sums."""
        return np.cumsum(np.cumsum(arr.astype(np.float64), axis=0), axis=1)

    def _patch_sums(self, integral, radius):
        """Return a 2-D array where each cell is the sum of the (2r+1)^2 patch
        centred at that cell, using the precomputed integral image."""
        rows, cols = integral.shape
        pad = np.zeros((rows + 1, cols + 1), dtype=np.float64)
        pad[1:, 1:] = integral

        r = radius
        xs = np.arange(rows)
        ys = np.arange(cols)

        x1 = np.clip(xs - r - 1, -1, rows - 1)
        x2 = np.clip(xs + r,      0, rows - 1)
        y1 = np.clip(ys - r - 1, -1, cols - 1)
        y2 = np.clip(ys + r,      0, cols - 1)

        A = pad[x1[:, None] + 1, y1[None, :] + 1]
        B = pad[x2[:, None] + 1, y1[None, :] + 1]
        C = pad[x1[:, None] + 1, y2[None, :] + 1]
        D = pad[x2[:, None] + 1, y2[None, :] + 1]

        return D - B - C + A

    def _find_establish_center(self):
        habitat = self.territory_map.habitat_quality
        rows, cols = habitat.shape
        radius = max(1, self.diameter // 4)

        # vectorised patch quality for every cell using integral image
        integral = self._integral_image(habitat)
        quality_grid = self._patch_sums(integral, radius)   # (rows, cols)

        valid = quality_grid >= self.min_quality             # (rows, cols) bool

        if not np.any(valid):
            return None

        existing_centers = np.array(
            [t["center"] for t in self.territory_map.get_territories().values()],
            dtype=float
        )

        min_distance = self.diameter / 4.0

        if len(existing_centers) > 0:
            xs = np.arange(rows, dtype=float)[:, None]
            ys = np.arange(cols, dtype=float)[None, :]
            dx = xs[None, :, :] - existing_centers[:, 0, None, None]
            dy = ys[None, :, :] - existing_centers[:, 1, None, None]
            dists = np.sqrt(dx * dx + dy * dy)
            too_close = np.any(dists < min_distance, axis=0)
            valid &= ~too_close

        if not np.any(valid):
            return None

        world_center = np.array([rows / 2.0, cols / 2.0])
        xs = np.arange(rows, dtype=float)[:, None]
        ys = np.arange(cols, dtype=float)[None, :]
        centrality_penalty = np.sqrt((xs - world_center[0])**2 + (ys - world_center[1])**2)

        scores = quality_grid - 0.05 * centrality_penalty
        scores[~valid] = -np.inf

        best = np.argmax(scores)
        return (int(best // cols), int(best % cols))

    def _decide_fledgling(self, ind, sex, natal_territory, fallback_center):
        territories = self.territory_map.get_territories()

        if natal_territory not in territories:
            return "disperse", None, fallback_center

        quality = self._territory_quality(natal_territory)
        is_high_quality = self._is_high_quality(quality)

        if sex == "female":
            if is_high_quality:
                # Requested 84-100% stay-home range.
                p_request = random.uniform(0.84, 1.0)
            else:
                p_request = 0.3
        else:
            # Male baseline from requested framework.
            p_disperse = 0.25 if is_high_quality else 0.4
            if random.random() < p_disperse:
                return "disperse", None, fallback_center
            return "request_subordinate", natal_territory, fallback_center

        if random.random() < p_request:
            return "request_subordinate", natal_territory, fallback_center
        return "disperse", None, fallback_center

    def _decide_subordinate(self, ind, sex, age, territory, fallback_center):
        territories = self.territory_map.get_territories()

        if territory not in territories:
            return "disperse", None, fallback_center

        current = territories[territory]
        vacancy_key = "primary_male" if sex == "male" else "primary_female"

        # Rule: immediate challenge for local primary vacancy.
        if current[vacancy_key] is None:
            return "compete_primary", territory, fallback_center

        relatedness = self._relatedness_to_primary_female(ind, territory)
        if relatedness < self.min_kinship:
            if random.random() < 0.8:
                return "disperse", None, fallback_center
            return "nothing", territory, fallback_center

        if age >= 8:
            if random.random() < 0.65:
                return "disperse", None, fallback_center
            return "nothing", territory, fallback_center

        quality = self._territory_quality(territory)
        if self._is_high_quality(quality):
            return "nothing", territory, fallback_center

        if random.random() < 0.35:
            return "disperse", None, fallback_center
        return "nothing", territory, fallback_center

    def _decide_floater(self, ind, sex, fallback_center):

        #floaters can take three actions -> ["compete_primary", "establish_territory", "request_subordinate"]

        #get all territories to check for vacancies or good options
        territories = self.territory_map.get_territories()

        vacancy_territory = self._find_vacancy(sex)
        if vacancy_territory is not None:
            return "compete_primary", vacancy_territory, fallback_center

        if sex == "male":
            center = self._find_establish_center()
            if center is not None:
                return "establish_territory", None, center

            # If no viable center exists, request subordinate in best territory.
            if len(territories) > 0:
                best_territory = self._choose_highest_quality_territory(list(territories.keys()))
                return "request_subordinate", best_territory, fallback_center
            return "nothing", None, fallback_center

        # Female floaters: request subordinate in high-quality territory.
        if len(territories) > 0:
            best_territory = self._choose_highest_quality_territory(list(territories.keys()))
            return "request_subordinate", best_territory, fallback_center

        return "nothing", None, fallback_center

    # --------------------------------------------------------------------------
    # PRIMARY DECISIONS
    # These are called by the simulation to resolve primary male/female choices
    # about subordinates and reproduction in their territory.
    # All rules derive from kin selection and ecological constraints logic.
    # --------------------------------------------------------------------------

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
