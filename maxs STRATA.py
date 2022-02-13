import numpy as np
import cvxpy as cp
import json
total_num_agents = 9
num_tasks = 3
num_agents_per_task = 3
num_traits = 3
expert_Q = np.zeros((total_num_agents,num_traits)) 

alpha_range = [0.2, 1]
beta_range = [0, 10]
gamma_range = [2, 6]

expert_Q_alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1], size=(total_num_agents, 1)) # Coverage
expert_Q_beta = np.random.uniform(low=beta_range[0], high=beta_range[1], size=(total_num_agents, 1))   # Capacity
expert_Q_gamma = np.random.uniform(low=gamma_range[0], high=gamma_range[1], size=(total_num_agents, 1))   # Speed
expert_Q[:, [0]] = expert_Q_alpha
expert_Q[:, [1]] = expert_Q_beta
expert_Q[:, [2]] = expert_Q_gamma

def generate_rand_X(num_tasks, total_num_agents):
    X = np.zeros((num_tasks,total_num_agents))
    idx = np.arange(total_num_agents)
    np.random.shuffle(idx)
    idx = idx.reshape((num_tasks,num_tasks))
    for i in range(num_tasks):
        X[[i,i,i],idx[i, :] ] += 1
    return X

def generate_rand_X_species(num_tasks, num_species, num_species_per_task):
    X = np.ones((num_tasks, num_species)) * num_species_per_task
    done = False
    count = 0
    while not done:
        idx = np.random.randint(low=0, high=3, size=(2,))
        if (np.sum(X[idx[0], :]) > 3) and (np.sum(X[:, idx[1]]) > 3) and (X[idx[0], idx[1]] > 0):
            X[idx[0], idx[1]] -= 1

        if np.all(np.sum(X, axis=0) == 3) and np.all(np.sum(X, axis=1) == 3):
            done = True
        if count > 100:
            X = np.ones((num_tasks, num_species)) * num_species_per_task
            count = 0
        count += 1

    return X

# Generate possible Q
def generate_Q_agent( Y_s, num_tasks, total_num_agents, noise=True):
    Q_sol = cp.Variable((total_num_agents, num_tasks))
    X = generate_rand_X( num_tasks, total_num_agents)
    mismatch = Y_s - cp.matmul(X, Q_sol)
    obj = cp.Minimize(cp.pnorm(mismatch, 2))
    opt_prob = cp.Problem(obj) 
    opt_prob.solve() 
    Q = Q_sol.value
    if noise:
        Q += (np.multiply(np.random.uniform(-0.25, 0.25, size=Q.shape), Q))
    return Q, X
     
def find_X_sol_agent(Y_s, Q, num_tasks, total_num_agents):
    X_sol = cp.Variable((num_tasks, total_num_agents), boolean=True)
    mismatch = Y_s - cp.matmul(X_sol, Q)
    obj = cp.Minimize(cp.pnorm(mismatch, 2))
    constraints = [cp.matmul(X_sol.T, np.ones([num_tasks, 1])) <= np.ones([total_num_agents , 1]), 
                    cp.matmul(X_sol, np.ones([total_num_agents, 1])) == 3*np.ones([num_tasks, 1])]
    opt_prob = cp.Problem(obj, constraints)
    opt_prob.solve()
    X_candidate = np.round(X_sol.value)
    return X_candidate

def find_X_sol_species(Y_s, Q, num_tasks, num_species, X_original=None):
    X_sol = cp.Variable((num_tasks, num_species), boolean=False)
    mismatch = Y_s - cp.matmul(X_sol, Q)
    obj = cp.Minimize(cp.pnorm(mismatch, 2))
    constraints = [cp.matmul(X_sol.T, np.ones([num_tasks, 1])) >= 3*np.ones([num_species , 1]), 
                    cp.matmul(X_sol, np.ones([num_tasks, 1])) >= 3*np.ones([num_species , 1]), 
                    cp.matmul(X_sol.T, np.ones([num_tasks, 1])) <= 3*np.ones([num_species , 1]), 
                    cp.matmul(X_sol, np.ones([num_tasks, 1])) <= 3*np.ones([num_species , 1]), 
                    X_sol >= 0]
    opt_prob = cp.Problem(obj, constraints)
    opt_prob.solve(solver=cp.CPLEX)
    X_candidate = np.round(X_sol.value)

    return X_candidate

# Generate possible Q
def generate_Q_species( Y_s, num_tasks, num_species, noise=False):
    Q_sol = cp.Variable((num_species, num_tasks))
    X = generate_rand_X_species( num_tasks, num_species,3)
    mismatch = Y_s - cp.matmul(X, Q_sol)
    obj = cp.Minimize(cp.norm(mismatch, 'fro'))
    constraints = [Q_sol >= 0.01, Q_sol[:, 0] >= 0.333, Q_sol[:,0] <= 0.5, Q_sol[:, 2] >= 1]
    opt_prob = cp.Problem(obj, constraints) 
    opt_prob.solve()
    Q = Q_sol.value

    return Q

save_dict = {}
num_target_teams = 20
expert_X = np.array([[3,0,0],
                    [0,3,0],
                    [0,0,3]])
expert_Q = np.array([[0.5,2,1],
                    [0.2,10,1],
                    [0.2,2,2]]) # was 0.2, 2, 5

Y_s = expert_X @ expert_Q

target_save_dict = {}
target_save_dict[0] = {}
target_save_dict[0]['X'] = expert_X.tolist()
target_save_dict[0]['Y'] = Y_s.tolist()
target_save_dict[0]['Q'] = expert_Q.tolist()

random_save_dict = {}
random_save_dict[0] = {}
random_save_dict[0]['X'] = expert_X.tolist()
random_save_dict[0]['Y'] = Y_s.tolist()
random_save_dict[0]['Q'] = expert_Q.tolist()

uniform_save_dict = {}
uniform_save_dict[0] = {}
uniform_save_dict[0]['X'] = expert_X.tolist()
uniform_save_dict[0]['Y'] = Y_s.tolist()
uniform_save_dict[0]['Q'] = expert_Q.tolist()

for i in range(num_target_teams):
    valid = False
    while not valid:
        Q = generate_Q_species(Y_s, num_tasks, 3)
        X = find_X_sol_species(Y_s, Q, num_tasks, 3)
        X_uniform = find_X_sol_species(np.mean(Y_s, axis=0) * np.ones((num_tasks, num_traits)),Q, num_tasks, 3)
        if np.all(np.sum(X, axis=0) == 3) and np.all(np.sum(X, axis=1) == 3):
            valid = True
    
    target_save_dict[i+1] = {}
    target_save_dict[i+1]['Q'] = Q.tolist()
    target_save_dict[i+1]['X'] = X.tolist()
    target_save_dict[i+1]['Y'] = (X@Q).tolist()

    random_save_dict[i+1] = {}
    random_save_dict[i+1]['Q'] = Q.tolist()
    random_save_dict[i+1]['X'] = generate_rand_X_species(3,3,3).tolist()
    random_save_dict[i+1]['Y'] = (X@Q).tolist()

    uniform_save_dict[i+1] = {}
    uniform_save_dict[i+1]['Q'] = Q.tolist()
    uniform_save_dict[i+1]['X'] = X_uniform.tolist()
    uniform_save_dict[i+1]['Y'] = (X_uniform@Q).tolist()

with open('assigned_teams.json', 'w') as outfile:
    json.dump(target_save_dict, outfile, indent=3, sort_keys=False)

with open('random_teams.json', 'w') as outfile:
    json.dump(random_save_dict, outfile, indent=3, sort_keys=False)

with open('uniform_teams.json', 'w') as outfile:
    json.dump(uniform_save_dict, outfile, indent=3, sort_keys=False)

​