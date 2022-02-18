import numpy as np
import function_utils
import random


# contains variables for a trait-task relationship   
class STRATA_relationship():
    """
    Class STRATA_relationship: variables for a trait-task relationship
    """

    def __init__(self, trait="trait_default", task="task_default", func=function_utils.function_linear, max_u=None, param_a=1, param_b=0, param_c=0):
        self.trait = trait  # STRATA_trait object
        self.task = task  # STRATA_task object
        self.func = func  # performance function type
        self.max_u = max_u if max_u is not None else trait.max_val  # maximum trait value
        self.param_a = param_a  # performance function parameter A
        self.param_b = param_b  # performance function parameter B
        self.param_c = param_c  # performance function parameter C
    
    def set(self, func=None, param_a=None, param_b=None, param_c=None):
        if func is not None:
            if not isinstance(func, function):
                self.func = func

    def set_function(self, func):
        self.func = func

    def set_max_u(self, max_u):
        self.max_u = max_u
    
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
    def generate_random_linear_relationship(self, max_u, min_m=None, max_m=None):
        # default to the task min/max
        if min_m is None:
            min_m = self.task.min_m
        if max_m is None:
            max_m = self.task.max_m

        self.func = function_utils.function_linear
        self.param_b = random.random() * (max_m - min_m) + min_m  # y shift
        self.param_a = random.random() * (max_m - min_m) / max_u - self.param_b / max_u  # slope
        self.param_c = 0

    # generate a random power relationship
    def generate_random_power_relationship(self, max_u=None, min_m=None, max_m=None):
        # param A: x scalar
        # param B: x exponent
        # param C: y shift
        self.func = function_utils.function_power
        self.param_b = random.randint(1, 10)  # x exponent
        
        # default to the task/trait's max/min u/m
        max_u = max_u if max_u is not None else self.max_u
        min_m = min_m if min_m is not None else self.task.min_m
        max_m = max_m if max_m is not None else self.task.max_m

        # if a max_u is specified, set it for the relationship
        self.max_u = max_u if max_u is not None else self.max_u
        
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
        if sign == 1:  # if positive a, choose y intercept of 0
            self.param_c = 0  # y intercept of 0
            max_u_intercept = random.random() * (max_m - self.param_c)  # select an intercept at the max u
            self.param_a = max_u_intercept / (max_u ** self.param_b)  # backcalculate param a
        if sign == -1:  # if negative a, choose random y intercept
            self.param_c = random.random() * (max_m - min_m) + min_m  # y intercept from min to max trait
            self.param_a = -1 * self.param_c / (max_u ** self.param_b)  # x scalar

    # calculate the relationship's task performance given the trait
    def calc_relationship_performance(self, x):
        x = min(self.max_u, x)
        p = self.func(x, self.param_a, self.param_b, self.param_c)
        return p
