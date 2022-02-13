# does STRATA optimization WITHOUT CONSIDERING THE K TERM (dynamic robot assignment). Code originally from Max Rudolph and Harish Ravichandar, adapted by Jack Kolb

import numpy as np
import cvxpy as cp
import json

# contains variables and methods for solving STRATA's task optimization without the dynamic reassignment (k) term, so effectively trait-matching optimization
class static_STRATA():

    # initialize STRATA
    def __init__(self, num_species=3, num_tasks=3, num_agents_per_task=3, num_traits=3, trait_ranges=None, total_num_agents=None):
        # set the environment parameters
        self.num_species = num_species
        self.num_tasks = num_tasks
        self.total_num_agents = total_num_agents if total_num_agents is not None else num_species * num_agents_per_task  # default num agents to be the max possible without forcing overflow
        self.num_agents_per_task = num_agents_per_task
        self.num_traits = num_traits
        self.trait_ranges = trait_ranges if trait_ranges is not None else [0, 1] * self.num_traits  # default traits to be 0-1

        # init the Q matrix to a S x U zeros matrix
        self.Q = np.zeros((num_species, num_traits))

        # init the Y_s matrix to a M x U zeros matrix
        self.Q = np.zeros((num_tasks, num_traits))

    # randomly assigns agents to tasks
    # output: matrix, M x num_agents x agent_assigned_to_task
    def generate_rand_X(self, num_tasks, total_num_agents):
        X = np.zeros((num_tasks,total_num_agents))  # create a M x N_r zeros matrix
        idx = np.arange(total_num_agents)  # randomly select agents in N_r by their index
        np.random.shuffle(idx)
        idx = idx.reshape((num_tasks,num_tasks))
        for i in range(num_tasks):  # for each task
            X[[i,i,i],idx[i, :] ] += 1  # add each agent to the task
        return X  # returns a wide M x num_robots matrix


    # randomly assigns species to tasks such that each species is using all possible robots and all tasks are fully assigned to
    # output: matrix, M x num_species x count 
    def generate_rand_X_species(self, num_tasks, num_species, num_species_per_task):
        # this function works by generating a saturated matrix and randomly cutting it down until the max robots per species and max robots per task constraints are satisfied
        X = np.ones((num_tasks, num_species)) * num_species_per_task  # begin by M x N_s matrix with all robots assigned to everything
        done = False
        count = 0
        while not done:  # for each step...
            idx = np.random.randint(low=0, high=3, size=(2,))  # choose two random indices: task and species
            # remove a robot from the selected task and species if:
            #   1. there are more robots of that task than can be possibly assigned (hardcoded to 3 in this case)
            #   2. there are more robots of that species that can possibly be assigned (also hardcoded to 3)
            #   3. there is at least one robot of that species assigned to that task
            # this keeps the number of robots per species and number of robots per task at or above their maximum values (both hardcoded to 3)
            if (np.sum(X[idx[0], :]) > 3) and (np.sum(X[:, idx[1]]) > 3) and (X[idx[0], idx[1]] > 0):
                X[idx[0], idx[1]] -= 1

            # if all robots of all species are maximally assigned to tasks, set done flag to true
            if np.all(np.sum(X, axis=0) == 3) and np.all(np.sum(X, axis=1) == 3):
                done = True

            # after 100 iterations, if we are still not done, reset and try again
            if count > 100:
                X = np.ones((num_tasks, num_species)) * num_species_per_task
                count = 0
            count += 1
        return X


    # Given the known optimal task trait matrix Y*, generates a random agent assignment X and finds X's optimal agent trait matrix Q to match Y*
    def generate_Q_agent(self, Y_s, num_tasks, total_num_agents, noise=True):
        # uses convex optimization to generate a trait matrix and assignment that fits the given Y*
        # in effect, this works backwards from Y* to find optimal Q for a given assignment
        Q_sol = cp.Variable((total_num_agents, num_tasks))  # define Q solution as a variable matrix of num_robots x M scalars
        X = self.generate_rand_X( num_tasks, total_num_agents)  # generate a random agent assignment as a starting point
        mismatch = Y_s - cp.matmul(X, Q_sol)  # find the mismatch between the optimal task traits and this random assignment x Q solution variable
        obj = cp.Minimize(cp.pnorm(mismatch, 2))  # set the optimization objective to minimize the pnorm (linear distance) of this mismatch, 2 indicates we are using square roots (a^2 + b^2 + c^2) ^ (1/2)
        opt_prob = cp.Problem(obj)  # set the optimization problem to this objective
        opt_prob.solve()  # solve the optimization
        Q = Q_sol.value  # get the value of the optimized Q
        if noise:  # if we are adding noise, uniformly vary the resulting traits
            Q += (np.multiply(np.random.uniform(-0.25, 0.25, size=Q.shape), Q))
        return Q, X


    # assigns individual agents to tasks to get X 
    def find_X_sol_agent(self, Y_s, Q, num_tasks, total_num_agents):
        X_sol = cp.Variable((num_tasks, total_num_agents), boolean=True)  # define X_sol as a variable matrix of M x num_robots BOOLEANS
        mismatch = Y_s - cp.matmul(X_sol, Q)  # represent the mismatch
        obj = cp.Minimize(cp.pnorm(mismatch, 2))  # set the objective to minimizing the pnorm of the mismatch
        constraints = [cp.matmul(X_sol.T, np.ones([num_tasks, 1])) <= np.ones([total_num_agents , 1]),   # add constraints: cannot exceed max agents per task
                        cp.matmul(X_sol, np.ones([total_num_agents, 1])) == 3*np.ones([num_tasks, 1])]
        opt_prob = cp.Problem(obj, constraints)  # load the optimization and constraints
        opt_prob.solve()  # solve
        X_candidate = np.round(X_sol.value)  # round the solution
        return X_candidate  # return the solution


    # assigns robots by species to tasks to get X
    def find_X_sol_species(self):
        X_sol = cp.Variable((self.num_tasks, self.num_species), boolean=False)  # define X_sol as a variable matrix of M x num_robots scalars
        mismatch = Y_s - cp.matmul(X_sol, Q)  # represent the mismatch
        obj = cp.Minimize(cp.pnorm(mismatch, 2))  # set the objective to minimizing the pnorm of the mismatch
        constraints = [cp.matmul(X_sol.T, np.ones([self.num_tasks, 1])) <= 3*np.ones([self.num_species , 1]),  # add constraints
                        cp.matmul(X_sol, np.ones([self.num_tasks, 1])) <= 3*np.ones([self.num_species , 1]), 
                        X_sol >= 0]
        opt_prob = cp.Problem(obj, constraints)  # load the optimization and constraints
        opt_prob.solve()  # solve
        X_candidate = np.round(X_sol.value)  # round the solution; NOTE: might want to change this to ensure robots aren't overassigned
        return X_candidate  # return the solution


    # generate an optimal Q that solves a given Y* with the given traits using a hidden assignment X; this only returns the trait matrix
    def generate_Q_species(self, Y_s, num_tasks, num_species, noise=False):
        Q_sol = cp.Variable((num_species, num_tasks))  # define Q as a variable trait matrix of S x M scalars, NOTE: this assumes # traits == # tasks
        X = self.generate_rand_X_species( num_tasks, num_species,3)  # generate a random agent assignment, this is the known optimal assignment
        mismatch = Y_s - cp.matmul(X, Q_sol)  # construct the Y* - Y_x mismatch
        obj = cp.Minimize(cp.norm(mismatch, 'fro'))  # set the objective to minimizing the Frobenius norm of the mismatch
        constraints = [Q_sol >= 0.01, Q_sol[:, 0] >= 0.333, Q_sol[:,0] <= 0.5, Q_sol[:, 2] >= 1]  # set the threshold constraints per task
        opt_prob = cp.Problem(obj, constraints)  # load the optimization and constraints
        opt_prob.solve()  # solve
        Q = Q_sol.value  # get the resulting trait matrix
        return Q  # return the trait matrix


# Jack's trials

# 1. set 3 traits, 3 tasks, 3 species, max 3 species/task, and diagonal-heavy trait matrix
STRATA = static_STRATA()  # init STRATA to default parameters

Y_s = np.array([[0.8, 0.1, 0.1],  # row: task, col: trait, val: trait threshold
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8]])

STRATA.Y_s = Y_s  # set Y_s threshold matrix

Q = np.array([[2, 0.1, 0.1],  # row: species, col: trait, val: trait score
              [0.1, 0.8, 0.1],
              [0.1, 0.1, 0.8]])

STRATA.Q = Q  # set Q trait matrix

X = STRATA.find_X_sol_species()  # find the species assignment

print("Resulting X", X)

# expert_X = np.array([[3, 0, 0],  # row: tasks, col: species, val: count
#                      [0, 3, 0],
#                      [0, 0, 3]])

# expert_Q = np.array([[0.5, 0.4, 0.1],  # row: species, col: trait, val: trait
#                      [0.2, 0.1, 0.2],
#                      [0.2, 0.2, 0.6]])


