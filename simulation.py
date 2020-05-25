import numpy as np
import scipy
import scipy.linalg


# Computes V * exp_wt * U.
# By construction the exponential of our matrices are always real-valued.
def Expm(V, exp_wt, U):
    return np.real(V.dot(np.diag(exp_wt)).dot(U))


# Computes the robot distribution after t.
def ComputeX(t_list, K, X_init):
    is_iterable = np.iterable(t_list)
    if not is_iterable:
      t_list = [t_list]
    num_nodes = X_init.shape[0]
    num_species = X_init.shape[1]
    X = np.zeros((len(t_list), num_nodes, num_species))
    # For each species, get transition matrix and calculate final distribution of robots.
    for s in range(num_species):
      Ks = K[:, :, s]
      x0 = X_init[:, s]
      # Perform eigen-decomposition to compute matrix exponential repeatedly.
      w, V = scipy.linalg.eig(Ks, right=True)
      U = scipy.linalg.inv(V)
      for i, t in enumerate(t_list):
        wt = w * t
        exp_wt = np.exp(wt)
        X[i, :, s] = Expm(V, exp_wt, U).dot(x0)
    if is_iterable:
      X[X < 0] = 0
      return X
    X = X[0, :, :]
    X[X < 0] = 0
    return X


# Computes the trait distribution after t.
def ComputeY(t_list, K, X_init, Q):
    return ComputeX(t_list, K, X_init).dot(Q)


# Simulates random transitions.
def SimulateX(max_time, K, X_init, dt=None):
    num_nodes = X_init.shape[0]
    num_species = X_init.shape[1]
    if dt is None:
      dt = 0.1 / np.max(K)  # Auto-scale step.
    # Pre-compute transition probabilities.
    P = []
    for s in range(num_species):
      Ks = K[:, :, s]
      P.append(scipy.linalg.expm(dt * Ks))

    X = X_init.copy()
    Xs = [X]
    t = 0
    ts = [0]
    while t < max_time:
      new_X = np.zeros_like(X)
      for s in range(num_species):
        for m in range(num_nodes):
          transition_probabilities = P[s][:, m]
          num_robots_in_m = X[m, s]
          choices = np.random.choice(num_nodes, num_robots_in_m, p=transition_probabilities)
          for n in range(num_nodes):
            new_X[n, s] += np.sum(choices == n)
      X = new_X
      Xs.append(X)
      t += dt
      ts.append(t)
    return np.stack(Xs, axis=0), np.array(ts)


def SimulateY(max_time, K, X_init, Q, dt=None):
    X, ts = SimulateX(max_time, K, X_init, dt)
    return X.dot(Q), ts


def ComputeRandomY(t_list, K, X_init, Q, var_Q):
    num_nodes = X_init.shape[0]
    num_traits = Q.shape[1]
    X = ComputeX(t_list, K, X_init)
    Y_mean = X.dot(Q)
    Y = np.empty_like(Y_mean)
    for t in range(len(t_list)):
        Y_var = np.multiply(X[t], X[t]).dot(var_Q)
        for i in range(num_nodes):
            for j in range(num_traits):
                Y[t][i][j] = np.sqrt(Y_var[i][j]) * np.random.randn() + Y_mean[t][i][j]
    return Y


def ComputeRandomY2(t_list, K, X_init, Q, var_Q):
    num_nodes = X_init.shape[0]
    num_species = X_init.shape[1]
    num_traits = Q.shape[1]
    X = ComputeX(t_list, K, X_init)
    Y = np.zeros([len(t_list), num_nodes, num_traits])
    for t in range(len(t_list)):
        Q_temp = np.empty_like(Q)
        for i in range(num_species):
            for j in range(num_traits):
                Q_temp[i][j] = np.sqrt(var_Q[i][j]) * np.random.randn() + Q[i][j]
        Y[t] = X[t].dot(Q_temp)
    return Y