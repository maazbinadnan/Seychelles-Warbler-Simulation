import random


class Population:
    _genome_keys = ("w_kin", "w_qual", "w_risk")

    def __init__(self, inds=[0,1], sex=['Male','Female']):
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
                "year": 0,
                "genome": self._random_genome(),
            }

    def _random_genome(self):
        return {key: random.random() for key in self._genome_keys}

    def _inherit_genome(self, father, mother):
        father_genome = self.pop_dict.get(father, {}).get("genome", self._random_genome())
        mother_genome = self.pop_dict.get(mother, {}).get("genome", self._random_genome())

        genome = {}
        for key in self._genome_keys:
            value = (father_genome[key] + mother_genome[key]) / 2
            if random.random() < 0.05:
                value += random.uniform(-0.05, 0.05)
            genome[key] = min(1.0, max(0.0, value))

        return genome

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
    def add(self, father, mother, ind, sex, fitness, territory, year, num_subordinates, quality, genome=None):
        self.pop_dict[father]["offspring"].append(ind)
        self.pop_dict[mother]["offspring"].append(ind)

        if genome is None:
            genome = self._inherit_genome(father, mother)
        else:
            genome = {key: min(1.0, max(0.0, genome.get(key, 0.5))) for key in self._genome_keys}

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
            "year": year,
            "genome": genome,
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

            else:  # primary
                actions = ["nothing"]

        else:  # female
            if life_history == "fledgling":
                actions = ["request_subordinate", "disperse"]

            elif life_history == "subordinate":
                actions = ["compete_primary", "disperse", "nothing"]

            elif life_history == "floater":
                actions = ["compete_primary", "request_subordinate", "nothing"]

            else:  # primary
                actions = ["nothing"]

        return actions
