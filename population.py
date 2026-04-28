class Population:
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
    def add(self, father, mother, ind, sex, fitness, territory, year, num_subordinates, quality):
        self.pop_dict[father]["offspring"].append(ind)
        self.pop_dict[mother]["offspring"].append(ind)
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
