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
import optimization_STRATA_plus
import simulation
# import continuous_trait_matrix
import stochastic_trait_matrix
import utils


# Choose problem parameters
num_tasks = 8
num_traits = 4
num_species = 5
robots_per_species = 200
max_rate = 2.  # maximum transition rate for the task graph

# TODO: make it possible to choose between i) task, ii) trait, and iii) task-trait preference
trait_weights = np.array([0.1, 0.4, 0.1, 0.4])  # 1 x num_tasks array to specify relative task priorities
task_trait_weights = np.array([[0.1, 0.4, 0.1, 0.4], [0.1, 0.4, 0.1, 0.4], [0.1, 0.4, 0.1, 0.4], [0.1, 0.4, 0.1, 0.4], [0.1, 0.4, 0.1, 0.4], [0.1, 0.4, 0.1, 0.4], [0.1, 0.4, 0.1, 0.4], [0.1, 0.4, 0.1, 0.4]])  # num_tasks x num_traits matrix to specify relative task and trait priorities

team_comp_flag = False  # flag to determine if team composition needs to be optimized

if team_comp_flag:
    num_tasks = num_tasks + 1  # add one node for idle task
    task_trait_weights = np.insert(task_trait_weights, 0, np.zeros(num_traits), axis=0)  # append a num_traits dimensional row of zeros for the idle task node


# Define task graph
g = graph.Graph(num_tasks)  # task graph. NOTE that the idle task (if it exists) will be assumed to be represented by the first node.
A = g.AdjacencyMatrix()  # task graph's adjacency matrix


# Trait preference experiment with multiple runs
num_exp_runs = 10
trait_err = np.zeros((num_exp_runs, num_traits))  # each row -> trait-wise proportional errors for corresponding run

for exp_run_ind in range(num_exp_runs):
    # Q = trait_matrix.CreateRandomQ(num_species, num_traits)
    Q, var_Q = stochastic_trait_matrix.CreateRankedQ(num_species, num_traits)  # team's capabilities
    # Define desired task requirements
    X_final = g.CreateRobotDistribution(num_species, robots_per_species, site_restrict=range(int(num_tasks/2), num_tasks))  # generate desired final configuration. The algorithm will NOT have access to this
    Y_desired = X_final.dot(Q)  # reverse engineer the desired trait distribution. The algorithm will have access to this
    if team_comp_flag:
        Y_desired[:, 0] = 0  # assign zeros to the requirements associated with the idle task.
        Y_desired = Y_desired * 0.9   # reduce requirements so that we have more resources than we need
    else:
        Y_desired = Y_desired * 1.1  # increase requirements so that we have less resources than we need

    # Choose simulation parameters
    num_simulations = 10
    num_simulations_rhc = 10
    rhc_steps = 10
    sim_rhc = False

    # initialize agent distribution
    X_init = g.CreateRobotDistribution(num_species, robots_per_species, site_restrict=range(0, int(num_tasks / 2)))  # inital agent distribution / assignment

    # optimize
    sys.stdout.write(utils.Highlight('Optimizing...\n', utils.BLUE, bold=True))
    sys.stdout.flush()
    # syntax: optimization.Optimize(Y_desired, A, X_init, Q, max_rate, warm_start_parameters=None, specified_time=None,
    # minimize_convergence_time=True, stabilize_robot_distribution=True, allow_trait_overflow=False, norm_order=2,
    # verify_gradients=False, verbose=False)
    K_opt, t_opt, _, _ = optimization_STRATA_plus.Optimize(Y_desired, A, X_init, Q, var_Q, task_trait_weights, max_rate, allow_trait_overflow=False, minimize_variance=True, analytical_gradients=True, verify_gradients=True, verbose=True, max_meta_iteration=50)
    sys.stdout.write(utils.Highlight('[OPTIMIZATION DONE]\n', utils.GREEN, bold=True))

    # Simulate (forward propagation)
    sys.stdout.write(utils.Highlight('[Simulating...]\n', utils.BLUE, bold=True))
    sys.stdout.flush()
    sim_time_steps = np.linspace(0., t_opt * 2., 100)  # simulate for twice as long as the optimal settling time
    Y_seq = simulation.ComputeY(sim_time_steps, K_opt, X_init, Q)  # time-series sequence of Y
    Y_ss = Y_seq[-1]  # steady-state Y
    sys.stdout.write(utils.Highlight('[SIMULATION DONE]\n', utils.GREEN, bold=True))
    # print('\n--------------\n')
    # print('Desired Y:\n', Y_desired)
    # print('\n')
    # print('Achieved Y:\n', Y_ss)

    # Measure Performance
    Y_diff = Y_desired - Y_ss  # element-wise difference ((+)value -> under-resourced; (-)value -> over-resourced)
    # print('\n')
    # print('Difference:\n', Y_diff)

    ind = 0
    for err_trait_i in Y_diff.T:
        err_trait_i[err_trait_i < 0] = 0  # no penalty for over-resourcing
        trait_err[exp_run_ind][ind] = np.sum(err_trait_i)/np.sum(Y_desired.T[ind])
        ind += 1

    print('\n')
    print('trait-wise error:', trait_err[exp_run_ind])

trait_err = np.nan_to_num(trait_err)
np.savetxt("trait_wise_error.csv", trait_err, delimiter=",")
