import numpy as np
import random
import json
import os

class qLearningAI():
    def _initialise_q_table(self):
        # seeding Q-table with artificial values so that q-values can mature 
        seeds = {
            # female floaters prefer joining territories
            ('floater', 'female', 'none', 'unrelated', 'young'): {
                'compete_primary': 8.0, 'request_subordinate': 7.0, 'nothing': 1.0
            },
            ('floater', 'female', 'none', 'unrelated', 'old'): {
                'compete_primary': 8.0, 'request_subordinate': 7.0, 'nothing': 1.0
            },
            #male floaters prefer establishing territory
            ('floater', 'male', 'none', 'unrelated', 'young'): {
                'establish_territory': 15.0, 'compete_primary': 6.0,
                'request_subordinate': 5.0, 'nothing': 1.0
            },
            ('floater', 'male', 'none', 'unrelated', 'old'): {
                'establish_territory': 8.0, 'compete_primary': 6.0,
                'request_subordinate': 5.0, 'nothing': 1.0
            },
            # fledglings prefer requesting subordinate on good territory
            ('fledgling', 'female', 'none', 'unrelated', 'young'): {
                'request_subordinate': 7.0, 'disperse': 2.0
            },
            ('fledgling', 'male', 'none', 'unrelated', 'young'): {
                'request_subordinate': 5.0, 'disperse': 4.0
            },
        }
        for state, actions in seeds.items():
            for action, value in actions.items():
                # Only seed if not already learned
                if (state, action) not in self.q_table:
                    self.q_table[(state, action)] = value

    def __init__(self, pop, territory_map, kinship, min_kinship, year, diameter, min_quality,
                 learning_rate=0.15, epsilon=0.3, q_table=None, q_table_path="output/q_table.json"):
        # initialise Q-learning model
        self.pop = pop
        self.territory_map = territory_map
        self.kinship = kinship
        self.min_kinship = min_kinship
        self.year = year
        self.diameter = diameter
        self.min_quality = min_quality

        # Q-learning parameters
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # Q-table with dict mapping (state_tuple, action_string) with float value
        # Shared across all birds and all episodes
        self.q_table_path = q_table_path
        self.q_table = q_table
        self.q_table = q_table if q_table is not None else {}
        self.load_q_table(self.q_table_path)
        self._initialise_q_table()

        # Tracks each individuals (state, action) pair from the current year
        # Used to update Q-values at the end of the year
        self.current_decisions = {}
        self.reward_accumulator = {}
        

    # ---------------------------
    # Interface wrappers
    # ---------------------------
    
    # save q_table to file for future use turning tuples to strings
    def save_q_table(self, path="output/q_table.json"):
        serialisable = {str(k): v for k, v in self.q_table.items()}
        with open(path, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"Q-table saved ({len(self.q_table)} entries) = {path}")

    # load previous Q-table from files converts string back to tuple or empty if no Q-table exists
    def load_q_table(self, path="output/q_table.json"):
        if not os.path.exists(path):
            print("No saved Q-table found, starting fresh.")
            return
        with open(path, "r") as f:
            raw = json.load(f)
        self.q_table = {eval(k): v for k, v in raw.items()}
        print(f"Q-table loaded ({len(self.q_table)} entries) ← {path}")
    #----------------------------------------
    # Helper functions same as utility based
    #----------------------------------------
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
    
    def _relatedness_to_primary_female(self, ind, territory):
        territories = self.territory_map.get_territories()
        primary_female = territories[territory]["primary_female"]

        if primary_female is None:
            return 0.0

        try:
            return float(self.kinship.calculate_relatedness(ind, primary_female))
        except Exception:
            return float(self.kinship.matrix.loc[ind, primary_female])
    
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
    
    def _territory_quality(self, territory_id):
        territories = self.territory_map.get_territories()
        if territory_id not in territories:
            return 0.0

        quality = territories[territory_id]["quality"]
        if quality is not None:
            return quality

        center = territories[territory_id]["center"]
        return self._local_quality(center, radius=max(1, self.diameter // 4))
    
    def _is_high_quality(self, quality):
        return quality >= (self.min_quality * 2.0)
    
    # gets current state of individual as a tuple of life_history, sex, quality band, kinship band and age band making them discrete
    
    def get_state(self, ind):
        
        life_history = self.pop[ind]["life_history"]
        sex = self.pop[ind]["sex"]
        territory_id = self.pop[ind]["territory"]
        territories = self.territory_map.get_territories()
        if territory_id is None or territory_id not in territories:
            quality_band = "none"
        else:
            quality = territories[territory_id]["quality"] or 0.0
            
            if quality < self.min_quality * 1.5:
                quality_band = "low"
            elif quality < self.min_quality * 3:
                quality_band = "medium"
            else:
                quality_band = "high"

            
        kinship_value = self._relatedness_to_primary_female(ind, territory_id) if territory_id in territories else 0.0
        kinship_band = "related" if kinship_value >= self.min_kinship else "unrelated"

        age = self.year - self.pop[ind]["year"]
        
        age_band = "young" if age <= 5 else "old"

        
        return (life_history, sex, quality_band, kinship_band, age_band)

     # chooses an action for individual using epsilon greedy policy
     # action is either random or the highest Q-value for the current state
     # and the choice is updated   
    def decide(self,ind):
        current_state = self.get_state(ind)
        valid_actions = self.pop.get_actions(ind)
        center = self._random_center()
        
        rand = random.random()
        
        if rand < self.epsilon:
            chosen_action = random.choice(valid_actions)
        else:
            q_values = {a: self.q_table.get((current_state, a), 0.0) for a in valid_actions}
            chosen_action = max(q_values, key=q_values.get)

        self.current_decisions[ind] = (current_state,chosen_action) 
        
        territory_id = self.pop[ind]["territory"]
        life_history = self.pop[ind]["life_history"]
        sex = self.pop[ind]["sex"]

        if chosen_action == "disperse":
            return "disperse", None, center

        elif chosen_action == "nothing":
            return "nothing", territory_id, center

        elif chosen_action == "compete_primary":
            if life_history == "floater":
                vacancy_key = "primary_female" if sex == "female" else "primary_male"
                territories = self.territory_map.get_territories()
                vacant = [t for t, info in territories.items() if info[vacancy_key] is None]
                target = random.choice(vacant) if vacant else territory_id
                return "compete_primary", target, center
            return "compete_primary", territory_id, center
            

        elif chosen_action == "request_subordinate":
            if life_history == "floater":
                territories = self.territory_map.get_territories()
                candidates = list(territories.keys())
                target = random.choice(candidates) if candidates else territory_id
                return "request_subordinate", target, center
            return "request_subordinate", territory_id, center

        elif chosen_action == "establish_territory":
            return "establish_territory", None, center

        return "nothing", territory_id, center       
    
    # 
    def update_q_values(self, ind, reward):
        if ind not in self.current_decisions:
            return
        state, action = self.current_decisions[ind]  # capture this year's state/action
        if ind not in self.reward_accumulator:
            self.reward_accumulator[ind] = []
        self.reward_accumulator[ind].append((state, action, reward))  # store all three
    
    # after an episode finishes the q-table is updated with accumulated rewards
    def end_of_episode_update(self):
        for ind, records in self.reward_accumulator.items():
            # Group rewards by (state, action) pair
            sa_rewards = {}
            for state, action, reward in records:
                key = (state, action)
                if key not in sa_rewards:
                    sa_rewards[key] = []
                sa_rewards[key].append(reward)

            # Update Q-value for each (state, action) using average reward
            for (state, action), rewards in sa_rewards.items():
                mean_reward = np.mean(rewards)
                current_q = self.q_table.get((state, action), 0.0)
                gamma = 0.9
                future_q = max(
                    (self.q_table.get((state, a), 0.0) for a in self.pop.get_actions(ind)),
                    default=0.0
                ) if ind in self.pop.get_dict() else 0.0
                self.q_table[(state, action)] = (
                    current_q + self.learning_rate * (mean_reward + gamma * future_q - current_q)
                )

        # Clear for next episode
        self.current_decisions = {}
        self.reward_accumulator = {}
        
    # same as in utility_based
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