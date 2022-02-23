# does STRATA optimization WITHOUT CONSIDERING THE K TERM (dynamic robot assignment). Code originally from Max Rudolph and Harish Ravichandar, adapted by Jack Kolb

import numpy as np
import numpy.linalg
#import cvxpy as cp
import json
import function_utils
import STRATA_task, STRATA_trait, STRATA_relationship, STRATA_species
import copy
import itertools
import random
import matplotlib.pyplot as plt


# contains variables and methods for solving STRATA's task optimization without the dynamic reassignment (k) term, so effectively trait-matching optimization
class static_STRATA():
    """
    Class static_STRATA: enables configuring tasks/robots and optimizes assignment.

    Variables:
        self.tasks          STRATA_task{}       dict of tasks
        self.traits         STRATA_trait{}      dict of traits
        self.species        STRATA_species[]    list of species

        self.Y_s            np.array(M,U)       optimal task-trait values
        self.Q              np.array(S,U)       species-trait values
        self.X              np.array(S,M)       species-task values of STRATA assignment
        self.X_complete     np.array(S,M)       species-task values of brute-force assignment
    
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
            trait = STRATA_trait.STRATA_trait(name="U" + str(n+1), distribution=function_utils.distribution_uniform, param_a=0, param_b=1)  
            self.traits[trait.name] = copy.deepcopy(trait)
        self.num_traits = num_traits
        print("... generated default traits")
        
        # generate tasks
        self.tasks = {}  # reset the task list
        for n in range(num_tasks):  # generate num_tasks new tasks
            task = STRATA_task.STRATA_task(name = "M" + str(n+1))
            task.generate_linear_relationships(self.traits)
            self.tasks[task.name] = copy.deepcopy(task)  # add the task to the task list
        self.num_tasks = num_tasks
        print("... generated default tasks")

        # generate species
        self.species = {}  # reset the species list
        for n in range(num_species):  # generate num_species new species
            # create a species with default sampled traits
            name = "S" + str(n+1)
            species = STRATA_species.STRATA_species(name=name, min_robots=0, max_robots=3)
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

    # generates randomized traits
    def generate_traits_random(self, max_u=1):
        for trait in self.traits:  # for each trait
            # if uniform distribution, randomize the trait range
            if self.traits[trait].distribution == function_utils.distribution_uniform:
                self.traits[trait].max_val = random.random() * max_u  # randomize the trait's max
            # if gaussian distribution, currently not implemented
            

    # generates randomized relationships
    def generate_relationships_random(self):
        for task in self.tasks:
            for trait in self.traits:
                self.get_relationship(task=task, trait=trait).generate_random_power_relationship(max_u=self.traits[trait].max_val)
        return

    # add a task
    def add_task(self, name=None, max_robots=3, trait_relationships={}):
        name = name if name is not None else "M" + str(self.num_tasks + 1)
        new_task = STRATA_task.STRATA_task(name=name, max_robots=0, trait_relationships=trait_relationships)
        self.tasks[name] = copy.deepcopy(new_task)
        self.num_tasks += 1
        return new_task

    # change a trait-task relationship
    def change_relationship(self, task=None, trait=None, function=None, max_u=None, param_a=None, param_b=None, param_c=None):
        if task is None:  # check if task is specified
            print("change_relationship(): must specify task name")
            return
        if trait is None:  # check if trait is specified
            print("change_relationship(): must specify trait name")
            return
        if function is not None:  # update function
            self.tasks[task].trait_relationships[trait].set_function(function)
        if max_u is not None:  # update max_u
            self.tasks[task].trait_relationships[trait].set_max_u(max_u)
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

    # get a relationship object given a task name and trait name
    def get_relationship(self, task=None, trait=None):
        # error checking
        if task is None:
            print("Missing \"task\" parameter.")
            return
        if task not in self.tasks:
            print("Task", "\"" + task + "\"", "does not exist!")
            return
        if trait is None:
            print("Missing \"trait\" parameter.")
            return
        if trait not in self.traits:
            print("Trait", trait, "does not exist!")
            return
        if trait not in self.tasks[task].trait_relationships:
            print("Trait", "\"" + trait + "\"", "does not exist for task", "\"" + task + "\"")
            return

        # get the relationship       
        return self.tasks[task].trait_relationships[trait]

    # get a task object given a task name
    def get_task(self, task=None):
        return self.tasks[task]

    # get a trait object give a task name
    def get_trait(self, trait=None):
        return self.traits[trait]

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

    # try every combination of robot distributions to find a solution, minimizing frobenius norm
    def solve_strata(self):
        best_error = float("inf")
        best_X = np.zeros((self.num_tasks, self.num_species))

        species_max = np.array([self.species[s].max_robots for s in sorted(list(self.species.keys()))])  # max robots per species for each species
        max_species_max = max(species_max)
        
        tasks_max = np.array([self.tasks[t].max_robots for t in sorted(list(self.tasks.keys()))])  # max robots per task for each task
        max_tasks_max = max(tasks_max)
        
        # get all possible X values -- from [0 to min(max_species_max, max_tasks_max)], array of num_tasks x num_traits
        X_possible = itertools.product(range(min(max_species_max, max_tasks_max)), repeat=self.num_tasks*self.num_traits)
        for X in X_possible:  # for each possible species/task assignment
            X = np.array(X)
            X = X.reshape((self.num_tasks, self.num_species))  # reshape into matrix

            # reject if too many robots assigned to a species
            if np.any(np.sum(X, axis=0) > species_max):
                continue

            # reject if too many robots assigned to a task
            if np.any(np.sum(X, axis=1) > tasks_max):
                continue
            
            # threshold traits and calculate the Frobenius norm
            traits = np.matmul(X, self.Q)
            thresholded_traits = np.minimum(self.Y_s, traits)  # threshold all traits to the Y_s (analytically calculated)
            error = np.linalg.norm(self.Y_s - thresholded_traits)  # calculate the error of the assignment, L2 norm (STRATA)

            if error < best_error:  # choose best assignment
                best_error = error
                best_X = X

        self.X = best_X
        best_p = self.calc_performance(X=best_X)

        return best_X, best_error, best_p

    # try every combination of robot distributions to find a solution to minimize performance difference
    def solve_performance(self, debug_X=None):
        p_opt, _ = self.calc_performance(optimal=True)  # use Y* to get the optimal performance
        best_p = {}
        best_error = float("inf")
        best_X = np.zeros((self.num_tasks, self.num_species))

        species_max = np.array([self.species[s].max_robots for s in sorted(list(self.species.keys()))])  # max robots per species for each species
        max_species_max = max(species_max)
        
        tasks_max = np.array([self.tasks[t].max_robots for t in sorted(list(self.tasks.keys()))])  # max robots per task for each task
        max_tasks_max = max(tasks_max)

        X_possible = itertools.product(range(min(max_species_max, max_tasks_max)), repeat=self.num_tasks*self.num_traits)
        for X in X_possible:
            X = np.array(X)
            X = X.reshape((3,3))
            
            # reject if too many robots assigned to a species
            if np.any(np.sum(X, axis=0) > species_max):
                continue

            # reject if too many robots assigned to a task
            if np.any(np.sum(X, axis=1) > tasks_max):
                continue
            
            # calculate the performance
            p, _ = self.calc_performance(X=X)
            error = self.calc_performance_error(p=p)

            # if debugging a specific assignment, print the assign's performance/error
            if debug_X is not None and np.all(X == debug_X):
                print("Debug Assignment: p", p, "best_p", best_p, "error", error, "best_error", best_error)

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

    # calculate task performance, specify optimal=True to use Y*, otherwise specify X to use Y=XQ
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

    # plots each trait/task relationship
    def plot_relationships(self, max_u=None):
        N = 1000  # number of sample points

        # generate the x and y storage matrices, each is task x trait
        x = []
        y = []
        for i_m in range(self.num_tasks):
            x.append([])
            y.append([])
            for i_u in range(self.num_traits):
                x[i_m].append([])
                y[i_m].append([])
        
        task_names = sorted(list(self.tasks.keys()))
        trait_names = sorted(list(self.traits.keys()))

        # sample across the task/trait axes
        for i_u in range(self.num_traits):  # for each trait
            for i_m in range(self.num_tasks):  # for each task
                for _ in range(N):
                    sample_trait = self.traits[trait_names[i_u]].sample()  # sample the trait value
                    val = self.get_relationship(task_names[i_m], trait_names[i_u]).calc_relationship_performance(sample_trait)  # get the relationship's value
                    x[i_m][i_u].append(sample_trait)  # add the value to the relationship
                    y[i_m][i_u].append(val)  # add the value to the relationship

        # plot the relationships
        fig, axs = plt.subplots(self.num_tasks, self.num_traits)  # create a subplot grid

        for i_u in range(self.num_tasks):
            for i_m in range(self.num_traits):
                axs[i_m, i_u].scatter(x[i_m][i_u], y[i_m][i_u], s=5)
                axs[i_m, i_u].set_title('Task ' + task_names[i_m] + ', Trait ' + trait_names[i_u])
                if max_u is not None:  # if a max u was not specified, default to the trait's max u
                    axs[i_m, i_u].set_xlim([self.traits[trait_names[i_u]].min_val, self.traits[trait_names[i_u]].max_val])
                else:
                    axs[i_m, i_u].set_xlim([0, max_u])
                axs[i_m, i_u].set_ylim([self.tasks[task_names[i_u]].min_m, self.tasks[task_names[i_u]].max_m])
                
                # hide label if not on edge
                axs[i_m, i_u].set_xlabel("Trait Value" if i_m == self.num_tasks - 1 else "")
                axs[i_m, i_u].set_ylabel("Task Value" if i_u == 0 else "")

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


