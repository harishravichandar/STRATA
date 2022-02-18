# does STRATA optimization WITHOUT CONSIDERING THE K TERM (dynamic robot assignment). Code originally from Max Rudolph and Harish Ravichandar, adapted by Jack Kolb

import numpy as np
import function_utils
import random

# contains variables for a task
class STRATA_task():
    """
    Class STRATA_task: defines a task
    """

    def __init__(self, name="task_default", min_robots=0, max_robots=3, trait_relationships={}):
        self.name = name
        self.min_robots = min_robots  # min robots that can be assigned to the task
        self.max_robots = max_robots  # max robots that can be assigned to the task
        self.trait_relationships = trait_relationships

    def set_name(self, name):
        self.name = name

    def set_max_robots(self, max_robots):
        self.max_robots = max_robots

    def set_trait_relationship(self, trait_name, trait_relationship):
        self.trait_relationships[trait_name] = trait_relationship

    def print(self):
        print("Task", self.name)
        print("  Max Robots:", self.max_robots)
        print("  Trait Relationships:")
        for trait in self.trait_relationships:
            self.trait_relationships[trait].print(offset=4)

    # calculate the performance of a task given trait values
    def calc_task_performance(self, traits):
        performance = 0
        for trait in traits:  # linearly add each trait's performance
            performance += self.trait_relationships[trait].calc_relationship_performance(traits[trait])
        return performance

    # generate linear trait relationships for the task
    def generate_linear_relationships(self, traits):
        for trait in traits:  # generate a relations  hip for each trait
            # create a linear 0-1 relationship for the trait-task
            relationship = STRATA_relationship(traits[trait].name, self.name, function_utils.function_linear, max_u=1, param_a=1, param_b=0)
            self.set_trait_relationship(traits[trait].name, relationship)  # add the relationship to the task


# contains variables for a trait
class STRATA_trait():
    """
    Class STRATA_trait: defines a trait
    """

    def __init__(self, name="trait_default", distribution=function_utils.distribution_uniform, min_val=0, max_val=1, param_a=1, param_b=0):
        self.name = name
        self.distribution = distribution  # trait sampling distribution
        
        self.min_val = min_val
        self.max_val = max_val

        self.param_a = param_a
        self.param_b = param_b
        

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


# contains variables for a trait-task relationship   
class STRATA_relationship():
    """
    Class STRATA_relationship: variables for a trait-task relationship
    """

    def __init__(self, trait="trait_default", task="task_default", func=function_utils.function_linear, max_u=1, param_a=1, param_b=0, param_c=0):
        self.trait = trait  # trait name
        self.task = task  # task name
        self.func = func  # performance function type
        self.max_u = max_u  # maximum trait value
        self.param_a = param_a  # performance function parameter A
        self.param_b = param_b  # performance function parameter B
        self.param_c = param_c  # performance function parameter C
    
    def set(self, func=None, param_a=None, param_b=None, param_c=None):
        if func is not None:
            if not isinstance(func, function):
                self.func = func

    def set_function(self, func):
        self.func = func
    
    def set_param_a(self, param_a):
        self.param_a = param_a

    def set_param_b(self, param_b):
        self.param_b = param_b

    def set_param_c(self, param_c):
        self.param_c = param_c
    
    def print(self, offset=0):
        print(" " * offset + "Relationship", self.task, ":", self.trait)
        print(" " * offset + "  function:", self.func.__name__)
        print(" " * offset + "  A:", self.param_a)
        print(" " * offset + "  B:", self.param_b)
        print(" " * offset + "  C:", self.param_c)

    # gets the x corresponding to the maximum value of a function
    def get_y(self, min_val, max_val):
        # linear case: max value if positive slope, min value if negative slope
        if self.func == function_utils.function_linear:
            return max_val if self.param_a > 0 else min_val
        
        # power case: max value if positive A and abs(max) >= abs(min), min value if positive A and abs(min) > abs(max), 0 if negative A
        elif self.func == function_utils.function_power:
            if self.param_a > 0:
                if abs(max_val) >= abs(min_val):
                    return max_val
                else:
                    return min_val
            else:
                return 0

        # otherwise: use arange
        else:
            x = np.arange(self.min_val, self.max_val, 0.01)  # create array of small-spaced values
            y = self.func(x, self.param_a, self.param_b, self.param_c)  # find all y's
            max_x = x[np.argmax(y)]  # find the x of the maximum y
            return max_x  # return that x
        return
    
    # generate a random linear relationship
    def generate_random_linear_relationship(self, max_u, min_m, max_m):
        self.func = function_utils.function_linear
        self.param_b = random.random() * (max_m - min_m) + min_m  # y shift
        self.param_a = random.random() * (max_m - min_m) / max_u - self.param_b / max_u  # slope
        self.param_c = 0

    # generate a random power relationship
    def generate_random_power_relationship(self, max_u, min_m, max_m):
        # param A: x scalar
        # param B: x exponent
        # param C: y shift
        self.func = function_utils.function_power
        self.param_b = random.randint(1, 10)  # x exponent

        # currently force the function to be 0-1
        self.param_a = 1
        self.param_c = 0
        return

        # select two points from min to max
        left_point = random.random() * (max_m - min_m) + min_m
        right_point = random.random() * (max_m - min_m) + min_m

        # y1 = A x1^B + C
        # A = (y1 - C) ^ (1/B) / x1
        # y2 = A x2^B + C
        # C = y2 - A x2^B
        # --
        # A = (y1 - y2 + A x2^B) ^ (1/B) / x1
        # (A x1)^B = y1 - y2 + A x2^B
        # (A x1)^B - A x2^B = y1 - y2
        # 

        sign = 1 if random.random() > 0.5 else -1
        if sign == 1:  # if positive a, choose random y intercept
            self.param_c = random.random() * (max_m - min_m)  # y intercept from min to max trait
            self.param_a = sign * random.random() * (max_m - self.param_c)  # x scalar
        if sign == -1:  # if negative a, choose random y intercept
            self.param_c = random.random() * (max_m - min_m)  # y intercept from min to max trait
            self.param_a = sign * random.random() * (self.param_c - min_m)  # x scalar

    # calculate the relationship's task performance given the trait
    def calc_relationship_performance(self, x):
        x = min(self.max_u, x)
        p = self.func(x, self.param_a, self.param_c, self.param_c)
        return p


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

