import numpy as np
import function_utils
import random


# contains variables for a species
class STRATA_species():
    """
    Class STRATA_species: variables for a species
    """

    def __init__(self, name="species_default", min_robots=0, max_robots=3, traits={}):
        self.name = name  # species name
        self.min_robots = min_robots  # min number of robots
        self.max_robots = max_robots  # max number of robots
        self.traits = traits  # trait dictionary name:value

    def set_min_robots(self, min_robots=0):
        self.min_robots = min_robots
    
    def set_max_robots(self, max_robots=3):
        self.max_robots = max_robots
    
    # sync this species' traits to given traits in the dict form name:STRATA_trait
    def set_traits(self, traits):
        for trait in traits:  # sync the given traits
            self.traits[trait] = traits[trait]

    # generate trait values for the species using the traits' sampling
    def generate_trait_values(self, traits):
        for trait in traits:  # gets a value for each trait
            self.traits[trait] = traits[trait].sample()

    # get this species' Q (trait) vector, returns a (1, U) numpy array
    def get_Q_vector(self):
        sorted_traits = sorted(list(self.traits.keys()))  # sort the traits dict keys so all species have same order
        Q_vec = np.zeros((1, len(sorted_traits)))  # initialize Q_vec to 0 vector
        for i in range(len(sorted_traits)):  # for each trait
            Q_vec[0,i] = self.traits[sorted_traits[i]]  # set the value
        return Q_vec

