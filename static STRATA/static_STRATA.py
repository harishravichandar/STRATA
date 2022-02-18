# does STRATA optimization WITHOUT CONSIDERING THE K TERM (dynamic robot assignment). Code originally from Max Rudolph and Harish Ravichandar, adapted by Jack Kolb

import numpy as np
import numpy.linalg
import cvxpy as cp
import json
import function_utils
import STRATA_subclasses
import copy
import itertools


# contains variables and methods for solving STRATA's task optimization without the dynamic reassignment (k) term, so effectively trait-matching optimization
class static_STRATA():
    """
    Class static_STRATA: enables configuring tasks/robots and optimizes assignment.

    Variables:
        self.tasks          STRATA_task[]       list of tasks
        self.traits         STRATA_trait{}      dict of traits
        self.species        STRATA_species[]    list of species

        self.Y_s            np.array(M,U)       optimal task-trait values
        self.Q              np.array(S,U)       species-trait values
        self.X              np.array(S,M)       species-task values
    
    Methods:
        self.generate_default_config            generates default all-linear tasks and traits
    """

    # initialize STRATA
    def __init__(self):
        # set the environment parameters
        self.tasks = {}
        self.traits = {}
        self.species = {}

        self.num_tasks = 0
        self.num_traits = 0
        self.num_species = 0

        self.Y_s = np.zeros((self.num_tasks, self.num_traits))  # init the Y_s matrix to a M x U zeros matrix
        self.Q = np.zeros((self.num_species, self.num_traits))  # init the Q matrix to a S x U zeros matrix
        self.X = np.zeros((self.num_tasks, self.num_species))  # init the X matrix to a M x S zeros matrix
        self.X_complete = np.zeros((self.num_tasks, self.num_species))  # init the X complete matrix to a M x S zeros matrix

        self.generate_default_config()  # generate default tasks
        
    # generates default traits, tasks, species (3 of each, linear 0-1 relationships)
    def generate_default_config(self, num_traits=3, num_tasks=3, num_species=3):
        # generate traits
        self.traits = {}  # reset the trait list
        for n in range(num_traits):
            # generate a linear trait from 0-1
            trait = STRATA_subclasses.STRATA_trait(name="U" + str(n+1), distribution=function_utils.distribution_uniform, param_a=0, param_b=1)  
            self.traits[trait.name] = copy.deepcopy(trait)
        self.num_traits = num_traits
        print("... generated default traits")
        
        # generate tasks
        self.tasks = {}  # reset the task list
        for n in range(num_tasks):  # generate num_tasks new tasks
            task = STRATA_subclasses.STRATA_task(name = "M" + str(n+1))
            task.generate_linear_relationships(self.traits)
            self.tasks[task.name] = copy.deepcopy(task)  # add the task to the task list
        self.num_tasks = num_tasks
        print("... generated default tasks")

        # generate species
        self.species = {}  # reset the species list
        for n in range(num_species):  # generate num_species new species
            # create a species with default sampled traits
            name = "S" + str(n+1)
            species = STRATA_subclasses.STRATA_species(name=name, min_robots=0, max_robots=3)
            species.generate_trait_values(self.traits)
            self.species[name] = copy.deepcopy(species)
        self.num_species = num_species
        print("... generated default species")

        # update the matrices
        self.get_Q()
        self.get_Y_s()
        return

    # flatline all trait-task relationships
    def generate_relationships_flatline(self):
        for task in self.tasks:
            for trait in self.traits:
                self.change_relationship(task=task, trait=trait, function=function_utils.function_linear, param_a=0, param_b=0)
        return

    # add a task
    def add_task(self, name=None, max_robots=3, trait_relationships={}):
        name = name if name is not None else "M" + str(self.num_tasks + 1)
        new_task = STRATA_subclasses.STRATA_task(name=name, max_robots=0, trait_relationships=trait_relationships)
        self.tasks[name] = copy.deepcopy(new_task)
        self.num_tasks += 1
        return new_task

    # change a trait-task relationship
    def change_relationship(self, task=None, trait=None, function=None, param_a=None, param_b=None, param_c=None):
        if task is None:  # check if task is specified
            print("change_relationship(): must specify task name")
            return
        if trait is None:  # check if trait is specified
            print("change_relationship(): must specify trait name")
            return
        if function is not None:  # update function
            self.tasks[task].trait_relationships[trait].set_function(function)
        if param_a is not None:  # update param_a
            self.tasks[task].trait_relationships[trait].set_param_a(param_a)
        if param_b is not None:  # update param_b
            self.tasks[task].trait_relationships[trait].set_param_b(param_b)
        if param_c is not None:  # update param_c
            self.tasks[task].trait_relationships[trait].set_param_c(param_c)

    # change a species
    def change_species(self, species=None, trait=None, trait_value=None, min_robots=None, max_robots=None):
        if species is None:  # check if species is specified
            print("change_species(): must specify species name")
            return
        if trait is not None and trait_value is None:  # check if trait is specified but not the value to update to
            print("change_species(): specified trait, but did not specify value to change to")
            return
        if trait_value is not None and trait is None:  # check if trait is not specified but a value is given
            print("change_species(): specified trait value, but did not specify trait to change")

        if trait is not None and trait_value is not None:  # if changing a trait value
            if trait not in self.species[species].traits:  # check if trait exists
                print("change_species(): specified trait does not exist for species", species, "!")
                return
            if not isinstance(trait_value, float) and not isinstance(trait_value, int):  # check if trait is a float or int
                print("change_species(): trait value must be a float or an int")
                return
            self.species[species].traits[trait] = trait_value  # update the trait value

        if min_robots is not None:  # if changing the min robots
            if not isinstance(min_robots, int):  # ensure min_robots is an integer
                print("change_species(): min_robots must be an integer")
                return
            if min_robots < 0:  # ensure min_robots is >= 0
                print("change_species(): min_robots must be >= 0")
                return
            self.species[species].min_robots = min_robots  # update min_robots

        if max_robots is not None:  # if changing the max robots
            if not isinstance(max_robots, int):  # ensure max_robots is an integer
                print("change_species(): max_robots must be an integer")
                return
            if max_robots < 0:  # ensure max_robots is >= 0
                print("change_species(): max_robots must be >= 0")
                return
            self.species[species].max_robots = max_robots  # update max_robots
        
    # get the Q matrix
    def get_Q(self):
        Q = np.zeros((self.num_species, self.num_traits))  # initialize a zero matrix
        for i, species in enumerate(sorted(list(self.species.keys()))):  # for each species sorted by name
            Q[i,:] = self.species[species].get_Q_vector()  # insert the species' Q vector into the matrix
        self.Q = Q
        return Q

    # get the Y_s matrix
    def get_Y_s(self):
        Y_s = np.zeros((self.num_tasks, self.num_traits))  # initialize a zero matrix
        # for each task, for each trait, find the optimal Y value
        sorted_tasks = sorted(list(self.tasks.keys()))
        sorted_traits = sorted(list(self.traits.keys()))
        for m in range(len(sorted_tasks)):  # for each task index
            for u in range(len(sorted_traits)):  # for each trait index
                # set the value of Y_s to their maximum value
                Y_s[m,u] = self.tasks[sorted_tasks[m]].trait_relationships[sorted_traits[u]].get_y(self.traits[sorted_traits[u]].min_val, self.traits[sorted_traits[u]].max_val)
        self.Y_s = Y_s
        return Y_s

    # updates the internal Q matrix and Y_s matrix
    def update(self):
        self.get_Q()
        self.get_Y_s()

    # assigns robots by species to tasks to get X
    def solve(self):
        X_sol = cp.Variable((self.num_tasks, self.num_species), boolean=False)  # define X_sol as a variable matrix of M x S integers
        mismatch = self.Y_s - cp.matmul(X_sol, self.Q)  # represent the mismatch
        obj = cp.Minimize(cp.pnorm(mismatch, 2))  # set the objective to minimizing the pnorm of the mismatch
        constraints = [cp.matmul(X_sol.T, np.ones([self.num_tasks, 1])) <= 3*np.ones([self.num_species , 1]),  # add constraints
                        cp.matmul(X_sol, np.ones([self.num_tasks, 1])) <= 3*np.ones([self.num_species , 1]), 
                        X_sol >= 0]
        opt_prob = cp.Problem(obj, constraints)  # load the optimization and constraints
        opt_prob.solve()  # solve
        #X_candidate = X_sol.value
        X_candidate = np.round(X_sol.value)  # round the solution; NOTE: might want to change this to ensure robots aren't overassigned
        self.X = X_candidate
        return X_candidate  # return the solution

    # try every combination of robot distributions to find a solution
    def complete_solve(self):
        p_opt, _ = self.calc_performance(optimal=True)  # use Y* to get the optimal performance
        best_p = {}
        best_error = float("inf")
        best_X = np.zeros((self.num_tasks, self.num_species))
        X_possible = itertools.product([0, 1, 2, 3],repeat=9)  # Note: REPLACE THIS WITH ONE THAT ALLOWS FOR MORE ROBOTS/TRAITS/TASKS/MAX_ROBOTS
        for X in X_possible:
            X = np.array(X)
            X = X.reshape((3,3))
            # reject if too many robots
            if np.any(np.sum(X, axis=0) > 3):
                continue
            
            # calculate the performance
            p, _ = self.calc_performance(X=X)
            error = self.calc_performance_error(p=p, p_opt=p_opt)

            if error < best_error:
                best_p = p
                best_X = X
                best_error = error
        self.X_complete = best_X
        return best_X, best_p

    # deconstruct Y to a task:trait dictionary
    def deconstruct_Y(self, Y):
        m_u = {}
        sorted_tasks = sorted(list(self.tasks.keys()))  # sort the task names
        sorted_traits = sorted(list(self.traits.keys()))  # sort the trait names
        for m in range(len(sorted_tasks)):  # for each task
            m_u[sorted_tasks[m]] = {}  # init the task
            for u in range(len(sorted_traits)):  # for each trait
                m_u[sorted_tasks[m]][sorted_traits[u]] = Y[m,u]  # set the task-trait to the Y_s spot
        return m_u  # return the task-trait dictionary

    # calculate task performance
    def calc_performance(self, X=None, optimal=False, debug=False):
        p = {}
        if not optimal:  # if not optimal, matrix multiply XQ to get Y
            if X is None:  # if X not defined, use self.X
                X = self.X
            Y = np.matmul(X, self.Q)
        else:  # otherwise use Y*
            Y = self.Y_s
        m_u = self.deconstruct_Y(Y)  # deconstruct Y to a dictionary form
        
        for task in m_u:  # for each task
            p[task] = self.tasks[task].calc_task_performance(m_u[task])  # note the task's performance
        return p, Y

    # calc MSE error between two performances
    def calc_performance_error(self, p=None, p_opt=None):
        if p_opt is None:
            p_opt, _ = self.calc_performance(self.Y_s)
        if p is None:
            print("calc_performance_error(): must specify a performance to compare to!")
            return 
        error = 0
        for task in self.tasks:  # calculate mean squared error
            error += (p_opt[task] - p[task]) ** 2
        error /= len(list(self.tasks.keys()))
        return error


    # print Y_s
    def print_Y(self, Y=None):
        Y = Y if Y is not None else self.Y_s
        self.matrixprint(Y, R="M", C="U", rows=sorted(list(self.tasks.keys())), cols=sorted(list(self.traits.keys())), digits=2)

    # print Q
    def print_Q(self):
        self.matrixprint(self.Q, R="S", C="U", rows=sorted(list(self.species.keys())), cols=sorted(list(self.traits.keys())), digits=2)

    # print X
    def print_X(self, X=None):
        X = X if X is not None else self.X
        self.matrixprint(X, R="M", C="S", rows=sorted(list(self.tasks.keys())), cols=sorted(list(self.species.keys())), digits=2)

    # nicely prints out a matrix
    def matrixprint(self, X, R="R", C="C", rows=None, cols=None, digits=1):
        # print header
        line = " "
        for col in range(X.shape[1]):
            line += "    " + (C + str(col) if cols is None else cols[col])
        print(line)

        # print rows
        for row in range(X.shape[0]):
            line = (R + str(row) if rows is None else rows[row]) + "   "
            for col in range(X.shape[1]):
                val = str(round(np.abs(X[row,col]), digits))
                line += val + " " * (6 - len(val))
            print(line)
        print()
        return


# computes the performance of an assignment X
def performance(X, Q):
    # use cubic root on everything
    Y = np.matmul(X, Q)  # find the resulting trait matrix M x U
    print("Y Assignment")
    prettyprint(Y, R="M", C="U", digits=2)
    #Y[0,:] = np.cbrt(Y[0,:])  # use cubic root for the first task (row 0)
    Y[0,:] = np.power(Y[0,:],3)  # use power for the first task (row 0)
    #Y[1,:] = np.cbrt(Y[1,:])  # use cubic root for the second task (row 1)
    #Y[2,:] = np.cbrt(Y[2,:])  # use cubic root for the third task (row 2)
    Y = np.multiply(Y, np.identity(Y.shape[0]))  # isolate only the diagonal traits
    p = np.sum(np.cbrt(Y))
    return p

