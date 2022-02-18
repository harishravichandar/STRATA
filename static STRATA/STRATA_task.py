import numpy as np
import function_utils
import random
import STRATA_relationship

# contains variables for a task
class STRATA_task():
    """
    Class STRATA_task: defines a task
    """

    def __init__(self, name="task_default", min_m=0, max_m=1, min_robots=0, max_robots=3, trait_relationships={}):
        self.name = name
        self.min_robots = min_robots  # min robots that can be assigned to the task
        self.max_robots = max_robots  # max robots that can be assigned to the task
        self.min_m = min_m  # min task score
        self.max_m = max_m  # max task score
        self.trait_relationships = trait_relationships

    def set_name(self, name):
        self.name = name

    def set_max_robots(self, max_robots):
        self.max_robots = max_robots

    def set_max_m(self, max_m):
        self.max_m = max_m

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
        performance = min(performance, self.max_m)
        return performance

    # generate linear trait relationships for the task
    def generate_linear_relationships(self, traits):
        for trait in traits:  # generate a relations  hip for each trait
            # create a linear 0-1 relationship for the trait-task
            relationship = STRATA_relationship.STRATA_relationship(traits[trait], self, function_utils.function_linear, max_u=1, param_a=1, param_b=0)
            self.set_trait_relationship(traits[trait].name, relationship)  # add the relationship to the task

