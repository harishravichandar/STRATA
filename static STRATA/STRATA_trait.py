import numpy as np
import function_utils
import random


# contains variables for a trait
class STRATA_trait():
    """
    Class STRATA_trait: defines a trait
    """

    def __init__(self, name="trait_default", distribution=function_utils.distribution_uniform, min_val=0, max_val=1, param_a=1, param_b=0):
        self.name = name
        self.distribution = distribution  # trait sampling distribution
        
        self.min_val = min_val  # max value of the sampling
        self.max_val = max_val  # min value of the sampling

        self.param_a = param_a  # distribution parameter A
        self.param_b = param_b  # distribution parameter B
    
    def change_trait(self, distribution=None, min_val=None, max_val=None, param_a=None, param_b=None):
        if distribution is not None:
            self.distribution = distribution
        if min_val is not None:
            self.min_val = min_val
        if max_val is not None:
            self.max_val = max_val
        if param_a is not None:
            self.param_a = param_a
        if param_b is not None:
            self.param_b = param_b
        return

    def set_name(self, name):
        self.name = name

    def set_distribution(self, distribution, min_val=0, max_val=1, param_a=1, param_b=0):
        self.distribution = distribution
        
        self.min_val = min_val
        self.max_val = max_val

        self.param_a = param_a
        self.param_b = param_b

    def print(self, offset=0):
        print("Trait", self.name)
        print("  Min:", self.min_val)
        print("  Max:", self.max_val)
        print("  Distribution:", self.distribution.__name__)
        print("    A:", self.param_a)
        print("    B:", self.param_b)

    # sample a trait from the set distribution
    def sample(self):
        return self.distribution(self.min_val, self.max_val, self.param_a, self.param_b)
