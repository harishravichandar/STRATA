import matplotlib.pyplot as plt
import numpy as np
import sys
try:
    import tqdm
    has_tqdm = True
except ImportError:
    has_tqdm = False
    pass
import graph
import optimization_STRATA
import simulation
# import continuous_trait_matrix
import stochastic_trait_matrix
import utils

# Choose problem parameters
num_tasks = 8
num_traits = 4
num_species = 4
robots_per_species = 100
max_rate = 2.
min_trait_matching = True  # allows trait overflow
minimize_variance = True


# Define task graph and team capabilities
g = graph.Graph(num_tasks)  # task graph
A = g.AdjacencyMatrix()  # task graph's adjacency matrix
Q, var_Q = stochastic_trait_matrix.CreateRankedQ(num_species, num_traits)  # team's capabilities


# Define desired task requirements
X_final = g.CreateRobotDistribution(num_species, robots_per_species, site_restrict=range(int(num_tasks/2), num_tasks))  # generate desired final configuration. The algorithm will NOT have access to this.
Y_desired = X_final.dot(Q)  # reverse engineer the desired trait distribution. The algorithm will have access to this.


# Initialize agent distribution
X_init = g.CreateRobotDistribution(num_species, robots_per_species, site_restrict=range(0, int(num_tasks/2)))


# Choose simulation parameters
num_simulations = 10
num_simulations_rhc = 10
rhc_steps = 10
sim_rhc = False


# Optimize
sys.stdout.write('Optimizing...\n')
sys.stdout.flush()

# syntax: optimization_STRATA.Optimize(Y_desired, A, X_init, Q, var_Q, max_rate, gamma=0, warm_start_parameters=None,
#              specified_time=None, minimize_convergence_time=True, stabilize_robot_distribution=True,
#              allow_trait_overflow=False, norm_order=2, analytical_gradients=True, verify_gradients=False,
#              minimize_variance=False, max_meta_iteration=200, max_error=1e3, verbose=False)
K_opt, t_opt, _, _ = optimization_STRATA.Optimize(Y_desired, A, X_init, Q, var_Q, max_rate,
                                               allow_trait_overflow=min_trait_matching,
                                               minimize_variance=minimize_variance, analytical_gradients=True,
                                               verify_gradients=True, verbose=True)

sys.stdout.write(utils.Highlight('[OPTIMIZATION DONE]\n', utils.GREEN, bold=True))


# Simulate the optimized system
sys.stdout.write(utils.Highlight('[Simulating...]\n', utils.BLUE, bold=True))
sys.stdout.flush()
sim_time_steps = np.linspace(0., t_opt * 2., 100)  # simulate for twice as long as the optimal settling time
Y_seq = simulation.ComputeY(sim_time_steps, K_opt, X_init, Q)  # time-series evolution of actual trait distribution
Y_ss = Y_seq[-1]  # steady-state Y
sys.stdout.write(utils.Highlight('[SIMULATION DONE]\n', utils.GREEN, bold=True))
print('\n--------------\n')
print('Desired Y:\n', Y_desired)
print('\n')
print('Achieved Y:\n', Y_ss)
print('\n--------------\n')


# [Insert code to save the results as needed]

