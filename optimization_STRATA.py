import scipy
import scipy.optimize
import scipy.linalg
import numpy as np
import sys
import warnings

from collections import Counter

import simulation
import utils


# Different optimization modes.
ABSOLUTE_AT_LEAST = 0    # Use absolute error (error only when there are traits missing).
ABSOLUTE_EXACT = 1       # Use absolute error (1-norm).
QUADRATIC_AT_LEAST = 2   # Use quadratic error (error only when there are traits missing).
QUADRATIC_EXACT = 3      # Use quadratic error (Frobenius norm).


def CheckGradient(f, x, h=1e-4, max_reldiff=1e-4):
    fx, grad = f(x)  # Evaluate function value at original point
    assert x.shape == grad.shape, 'Variable and gradient must have the same shape'
    passed = True
    numerical_grad = np.empty_like(x)
    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        x[ix] -= h
        y1 = f(x)[0]
        x[ix] += 2 * h
        y2 = f(x)[0]
        numgrad = (y2 - y1) / (2 * h)
        x[ix] -= h
        numerical_grad[ix] = numgrad
        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > max_reldiff and passed:
            # Only print the first error.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', np.ComplexWarning)  # Ignore complex warning.
                print(utils.Highlight('Gradient check failed!', utils.RED, bold=True))
                print('First gradient error found at index %s' % str(ix))
                print('Your gradient: %f \t Numerical gradient: %f' % (grad[ix], numgrad))
            passed = False
        it.iternext()  # Step to next dimension.
    if passed:
        print(utils.Highlight('Gradient check passed!', utils.GREEN, bold=True))
    return numerical_grad


# To give feedback during the optimization.
current_iteration = 0
meta_iteration = 0

def Print(x, f, accepted):
    global current_iteration
    current_iteration += 1
    output = '\rIter: %i, Cost: %.4f, Accepted: %s' % (current_iteration, f, str(accepted))
    sys.stdout.write(output)
    # sys.stdout.write(' ' * (utils.GetTerminalWidth() - len(output))) # this works only in Linux!
    sys.stdout.flush()


# Computes V * exp_wt * U.
# By construction the exponential of our matrices are always real-valued.
def Expm(V, exp_wt, U):
    return np.real(V.dot(np.diag(exp_wt)).dot(U))


# Computes the mixture of costs.
#
# || Y(t) - Y_desired ||_2^2 + \alpha * t^2 + \beta * || X(t) - X(t + \nu) ||_2^2
#
# When setting \alpha to zero, the time t needs to be specified (through specified_time).
# When \beta is greater than zero, \nu needs to be specified.
#
# This function outputs both the cost and its Jacobian w.r.t. to the flattened rates (parameters)
# and time (when \alpha > 0).
#
# When the mode is *_AT_LEAST, the margin needs to be specified. The cost will penalize only missing traits (up to a margin).
# When the mode is QUADRATIC_*, the distance to Y_desired is measured quadratically.
# When the mode is ABSOLUTE_*, the distance to Y_desired is measured with the absolute value.
def Cost(parameters, Y_desired, A, X_init, Q, var_Q, alpha=1.0, specified_time=None, beta=5.0, gamma=1.0, nu=1.0,
         mode=QUADRATIC_EXACT, margin=None):
    # Sanity checks.
    assert alpha >= 0.0 and beta >= 0.0
    assert (alpha == 0.0) == (specified_time is not None)
    assert (beta == 0.0) == (nu is None)
    assert (mode in (QUADRATIC_EXACT, ABSOLUTE_EXACT)) == (margin is None)

    # Prepare variable depending on whether t part of the parameters.
    num_nodes = A.shape[0]
    num_species = X_init.shape[1]
    num_traits = Q.shape[1]
    if specified_time is None:
        t = parameters[-1]
        num_parameters_i = int((np.size(parameters) - 1) / num_species)
        grad_all = np.zeros(np.size(parameters))
    else:
        t = specified_time
        num_parameters_i = int(np.size(parameters) / num_species)
        grad_all = np.zeros(np.size(parameters))

    # Reshape adjacency matrix to make sure.
    Adj = A.astype(float).reshape((num_nodes, num_nodes))
    Adj_flatten = Adj.flatten().astype(bool)  # Flatten boolean version.

    # Loop through the species to compute the cost value.
    # At the same time, prepare the different matrices.
    Ks = []                     # K_s
    Z_0 = []                    # Z^{(s)}(0)
    eigenvalues = []            # w
    eigenvectors = []           # V.T
    eigenvectors_inverse = []   # U.T
    exponential_wt = []         # exp(eigenvalues * t).
    x_matrix = []               # Pre-computed X matrices.
    x0s = []                    # Avoids reshaping.
    qs = []                     # Avoids reshaping.
    xts = []                    # Keeps x_s(t).
    inside_norm = np.zeros((num_nodes, num_traits))  # Will hold the value prior to using the norm.
    for s in range(num_species):
        x0 = X_init[:, s].reshape((num_nodes, 1))
        z_0 = np.zeros(X_init.shape)
        z_0[:, s] = x0.reshape(num_nodes)
        Z_0.append(z_0)
        q = Q[s, :].reshape((1, num_traits))
        x0s.append(x0)
        qs.append(q)
        k_ij = parameters[s * num_parameters_i:(s + 1) * num_parameters_i]
        # Create K from individual k_{ij}.
        K = np.zeros(Adj_flatten.shape)
        K[Adj_flatten] = k_ij
        K = K.reshape((num_nodes, num_nodes))
        np.fill_diagonal(K, -np.sum(K, axis=0))
        # Store K.
        Ks.append(K)
        # Perform eigen-decomposition to compute matrix exponential.
        w, V = scipy.linalg.eig(K, right=True)
        U = scipy.linalg.inv(V)
        wt = w * t
        exp_wt = np.exp(wt)
        xt = Expm(V, exp_wt, U).dot(x0)
        inside_norm += xt.dot(q)
        # Store the transpose of these matrices for later use.
        eigenvalues.append(w)
        eigenvectors.append(V.T)
        eigenvectors_inverse.append(U.T)
        exponential_wt.append(exp_wt)
        xts.append(xt)
        # Pre-build X matrix.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # We don't care about 0/0 on the diagonal.
            X = np.subtract.outer(exp_wt, exp_wt) / (np.subtract.outer(wt, wt) + 1e-10)
        np.fill_diagonal(X, exp_wt)
        x_matrix.append(X)
    inside_norm -= Y_desired

    # Compute the trait mismatch cost depending on mode.
    derivative_outer_norm = None  # Holds the derivative of inside_norm (except the multiplication by (x0 * q)^T).
    if mode == ABSOLUTE_AT_LEAST:
        derivative_outer_norm = -inside_norm + margin
        value = np.sum(np.maximum(derivative_outer_norm, 0))
        derivative_outer_norm = -(derivative_outer_norm > 0).astype(float)  # Keep only 1s for when it's larger than margin.
    elif mode == ABSOLUTE_EXACT:
        abs_inside_norm = np.abs(inside_norm)
        index_zeros = abs_inside_norm < 1e-10
        value = np.sum(np.abs(inside_norm))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # We don't care about 0/0.
            derivative_outer_norm = inside_norm / abs_inside_norm  # Keep only 1s for when it's larger than 0 and -1s for when it's lower.
        derivative_outer_norm[index_zeros] = 0  # Make sure we set 0/0 to 0.
    elif mode == QUADRATIC_AT_LEAST:
        derivative_outer_norm = -inside_norm + margin
        value = np.sum(np.square(np.maximum(derivative_outer_norm, 0)))
        index_negatives = derivative_outer_norm < 0
        derivative_outer_norm *= -2.0
        derivative_outer_norm[index_negatives] = 0  # Don't propagate gradient on negative values.
    elif mode == QUADRATIC_EXACT:
        value = np.sum(np.square(inside_norm))
        derivative_outer_norm = 2.0 * inside_norm

    # compute cost to minimize time (if desired)
    value += alpha * (t ** 2)

    # Calculate gradient w.r.t. the transition matrix for each species
    for s in range(num_species):
        # Build gradient w.r.t. inside_norm of cost.
        top_grad = np.dot(derivative_outer_norm, np.dot(x0s[s], qs[s]).T)
        # Build gradient w.r.t. Exp(K * t).
        middle_grad = eigenvectors_inverse[s].dot(eigenvectors[s].dot(top_grad).dot(eigenvectors_inverse[s]) *
                                                  x_matrix[s]).dot(eigenvectors[s])
        # Build gradient w.r.t. K
        bottom_grad = middle_grad * t
        # Finally, propagate gradient to individual k_ij.
        grad = bottom_grad - np.diag(bottom_grad)
        grad = grad.flatten()[Adj_flatten]  # Reshape.
        grad_all[s*num_parameters_i:(s+1)*num_parameters_i] += np.array(np.real(grad))
        # Build gradient w.r.t. t (if desired)
        if specified_time is None:
            grad_all[-1] += np.real(np.sum(Ks[s] * middle_grad))

    # Gradient of alpha * t^2 w.r.t. t
    if specified_time is None:
        grad_all[-1] += 2.0 * t * alpha

    # Forcing the steady state.
    # We add a cost for keeping X(t) and X(t + nu) the same. We use the quadratic norm for this sub-cost.
    # The larger beta and the larger nu, the closer to steady state.
    if beta > 0.0:
        for s in range(num_species):
            # Compute exp of the eigenvalues of K * (t + nu).
            wtdt = eigenvalues[s] * (t + nu)
            exp_wtdt = np.exp(wtdt)
            # Compute x_s(t) - x_s(t + nu) for that species.
            # Note that since we store V.T and U.T, we do (U.T * D * V.T).T == V * D * U
            inside_norm = xts[s] - Expm(eigenvectors_inverse[s], exp_wtdt, eigenvectors[s]).T.dot(x0s[s])
            # Increment value.
            value += beta * np.sum(np.square(inside_norm))

            # Compute gradient on the first part of the cost: e^{Kt} x0 (we use the same chain rule as before).
            top_grad = 2.0 * beta * np.dot(inside_norm, x0s[s].T)
            store_inner_product = eigenvectors[s].dot(top_grad).dot(eigenvectors_inverse[s])  # Store to re-use.
            middle_grad = eigenvectors_inverse[s].dot(store_inner_product * x_matrix[s]).dot(eigenvectors[s])
            bottom_grad = middle_grad * t
            grad = bottom_grad - np.diag(bottom_grad)
            grad = grad.flatten()[Adj_flatten]  # Reshape.
            grad_all[s * num_parameters_i:(s + 1) * num_parameters_i] += np.array(np.real(grad))
            if specified_time is None:
                grad_all[-1] += np.real(np.sum(Ks[s] * middle_grad))

            # Compute gradient on the second part of the cost: e^{K(t + nu)} x0 (we use the same chain rule as before).
            # Compute X for e^{K(t + nu)}.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)  # We don't care about 0/0 on the diagonal.
                X = np.subtract.outer(exp_wtdt, exp_wtdt) / (np.subtract.outer(wtdt, wtdt) + 1e-10)
            np.fill_diagonal(X, exp_wtdt)
            # top_grad = 2.0 * beta * np.dot(inside_norm, x0s[s].T) [same as before but needs to be negated].
            middle_grad = -eigenvectors_inverse[s].dot(store_inner_product * X).dot(eigenvectors[s])
            bottom_grad = middle_grad * (t + nu)
            grad = bottom_grad - np.diag(bottom_grad)
            grad = grad.flatten()[Adj_flatten]  # Reshape.
            grad_all[s * num_parameters_i:(s + 1) * num_parameters_i] += np.array(np.real(grad))
            if specified_time is None:
                grad_all[-1] += np.real(np.sum(Ks[s] * middle_grad))

    # Minimize variance
    if gamma > 0.0:
        # Compute the variance cost
        X_t = np.squeeze(np.asarray(xts)).T  # convert the list to an array
        var_Y = np.dot(np.multiply(X_t, X_t), var_Q)  # compute Var(Y)
        value += gamma * np.sum(np.square(var_Y))  # add the square of the Frobenious norm of Var(Y) to the cost
        # derivative_outer_norm = 2.0 * gamma * var_Y

        # Calculate gradient of third part of the cost w.r.t. the transition matrix for each species
        for s in range(num_species):
            # Build gradient w.r.t. inside_norm of variance.
            # top_grad = 2.0 * np.dot(derivative_outer_norm, np.dot(np.multiply(Z_0[s], X_t), var_Q).T)
            top_grad = 4 * gamma * np.dot(np.multiply(np.dot(var_Y, var_Q.T), X_t), Z_0[s].T)
            # Build gradient w.r.t. Exp(K * t).
            middle_grad = eigenvectors_inverse[s].dot(
                    eigenvectors[s].dot(top_grad).dot(eigenvectors_inverse[s]) * x_matrix[s]).dot(eigenvectors[s])
            # Build gradient w.r.t. K
            bottom_grad = middle_grad * t
            # Finally, propagate gradient to individual k_ij.
            grad = bottom_grad - np.diag(bottom_grad)
            grad = grad.flatten()[Adj_flatten]  # Reshape.
            grad_all[s * num_parameters_i:(s + 1) * num_parameters_i] += np.array(np.real(grad))
            # Build gradient w.r.t. t (if desired)
            if specified_time is None:
                grad_all[-1] += np.real(np.sum(Ks[s] * middle_grad))

        """
        # BEGIN OLD CODE 
        
        grad_xts_t = np.zeros(np.shape(X_t))
        for s in range(num_species):
            # Compute gradients of variance cost w.r.t. each transition rate
            wt = eigenvalues[s].real * t
            exp_wt = np.exp(wt)
            mat_1 = np.diag(exp_wt) * t
            temp_mat_2 = np.tile(exp_wt, [num_nodes, 1])
            temp_mat_3 = np.tile(eigenvalues[s].real, [num_nodes, 1])
            mat_2 = temp_mat_2.T - temp_mat_2
            mat_3 = (temp_mat_3.T - temp_mat_3)
            np.fill_diagonal(mat_3, np.ones(num_nodes))

            temp_idx = mat_3[mat_3 == 0]
            if temp_idx.size:
                flag = 1

            mat_3 = 1 / mat_3
            np.fill_diagonal(mat_3, np.zeros(num_nodes))

            grad_norm_varY_K = np.zeros((num_nodes, num_nodes))
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if Adj[i][j]:
                        G_ij = np.outer(eigenvectors_inverse[s][i].real, eigenvectors[s].T[j].real)
                        V_ij = np.multiply(G_ij, (mat_1 + np.multiply(mat_2, mat_3)))
                        grad_xts_K_ij = np.dot(np.dot(np.dot(eigenvectors[s].T.real, V_ij),  
                                                eigenvectors_inverse[s].T.real),   Z_0[s])
                        grad_varY_K_ij = 2 * np.dot(np.multiply(grad_xts_K_ij, X_t), var_Q)
                        grad_norm_varY_K[i][j] = 2 * gamma * np.trace(np.dot(var_Y.T, grad_varY_K_ij)).real

            grad_norm_varY_K = grad_norm_varY_K.flatten()[Adj_flatten]  # Reshape.
            grad_all[s * num_parameters_i:(s + 1) * num_parameters_i] += np.array(np.real(grad_norm_varY_K))

            # Compute the gradient each species' distribution of w.r.t. time
            if specified_time is None:
                grad_xts_t += np.dot(np.multiply(Expm(eigenvectors_inverse[s].real, exp_wt, eigenvectors[s].real).T, 
                                                Ks[s]), Z_0[s])

        # Compute gradients of variance cost w.r.t. time
        if specified_time is None:
            grad_varY_t = 2 * np.dot(np.multiply(grad_xts_t, X_t), var_Q)
            grad_all[-1] += 2 * gamma * np.trace(np.dot(var_Y.T, grad_varY_t))
        
        # END OLD CODE
        """

    return [value, grad_all]


def cost_without_grad(parameters,
         Y_desired,
         A, X_init, Q, var_Q,
         alpha=1.0, specified_time=None,
         beta=5.0, gamma=1.0, nu=1.0,
         mode=QUADRATIC_EXACT, margin=None):
    # Sanity checks.f
    assert alpha >= 0.0 and beta >= 0.0
    assert (alpha == 0.0) == (specified_time is not None)
    assert (beta == 0.0) == (nu is None)
    assert (mode in (QUADRATIC_EXACT, ABSOLUTE_EXACT)) == (margin is None)

    # Prepare variable depending on whether t part of the parameters.
    num_nodes = A.shape[0]
    num_species = X_init.shape[1]
    num_traits = Q.shape[1]
    if specified_time is None:
        t = parameters[-1]
        num_parameters_i = int((np.size(parameters) - 1) / num_species)
        grad_all = np.zeros(np.size(parameters))
    else:
        t = specified_time
        num_parameters_i = int(np.size(parameters) / num_species)
        grad_all = np.zeros(np.size(parameters))

    # Reshape adjacency matrix to make sure.
    Adj = A.astype(float).reshape((num_nodes, num_nodes))
    Adj_flatten = Adj.flatten().astype(bool)  # Flatten boolean version.

    # Loop through the species to compute the cost value.
    # At the same time, prepare the different matrices.
    Ks = []                     # K_s
    Z_0 = []                    # Z^{(s)}(0)
    eigenvalues = []            # w
    eigenvectors = []           # V.T
    eigenvectors_inverse = []   # U.T
    exponential_wt = []         # exp(eigenvalues * t).
    x_matrix = []               # Pre-computed X matrices.
    x0s = []                    # Avoids reshaping.
    qs = []                     # Avoids reshaping.
    xts = []                    # Keeps x_s(t).
    inside_norm = np.zeros((num_nodes, num_traits))  # Will hold the value prior to using the norm.
    for s in range(num_species):
        x0 = X_init[:, s].reshape((num_nodes, 1))
        z_0 = np.zeros(X_init.shape)
        z_0[:, s] = x0.reshape(num_nodes)
        Z_0.append(z_0)
        q = Q[s, :].reshape((1, num_traits))
        x0s.append(x0)
        qs.append(q)
        k_ij = parameters[s * num_parameters_i:(s + 1) * num_parameters_i]
        # Create K from individual k_{ij}.
        K = np.zeros(Adj_flatten.shape)
        K[Adj_flatten] = k_ij
        K = K.reshape((num_nodes, num_nodes))
        np.fill_diagonal(K, -np.sum(K, axis=0))
        # Store K.
        Ks.append(K)
        # Perform eigen-decomposition to compute matrix exponential.
        w, V = scipy.linalg.eig(K, right=True)
        U = scipy.linalg.inv(V)
        wt = w * t
        exp_wt = np.exp(wt)
        xt = Expm(V, exp_wt, U).dot(x0)
        inside_norm += xt.dot(q)
        # Store the transpose of these matrices for later use.
        eigenvalues.append(w)
        eigenvectors.append(V.T)
        eigenvectors_inverse.append(U.T)
        exponential_wt.append(exp_wt)
        xts.append(xt)
        # Pre-build X matrix.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # We don't care about 0/0 on the diagonal.
            X = np.subtract.outer(exp_wt, exp_wt) / (np.subtract.outer(wt, wt) + 1e-10)
        np.fill_diagonal(X, exp_wt)
        x_matrix.append(X)
    inside_norm -= Y_desired

    # Compute the final cost value depending on mode.
    derivative_outer_norm = None  # Holds the derivative of inside_norm (except the multiplication by (x0 * q)^T).
    if mode == ABSOLUTE_AT_LEAST:
        derivative_outer_norm = -inside_norm + margin
        value = np.sum(np.maximum(derivative_outer_norm, 0))
        derivative_outer_norm = -(derivative_outer_norm > 0).astype(float)  # Keep only 1s for when it's larger than margin.
    elif mode == ABSOLUTE_EXACT:
        abs_inside_norm = np.abs(inside_norm)
        index_zeros = abs_inside_norm < 1e-10
        value = np.sum(np.abs(inside_norm))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # We don't care about 0/0.
            derivative_outer_norm = inside_norm / abs_inside_norm  # Keep only 1s for when it's larger than 0 and -1s for when it's lower.
        derivative_outer_norm[index_zeros] = 0  # Make sure we set 0/0 to 0.
    elif mode == QUADRATIC_AT_LEAST:
        derivative_outer_norm = -inside_norm + margin
        value = np.sum(np.square(np.maximum(derivative_outer_norm, 0)))
        index_negatives = derivative_outer_norm < 0
        derivative_outer_norm *= -2.0
        derivative_outer_norm[index_negatives] = 0  # Don't propagate gradient on negative values.
    elif mode == QUADRATIC_EXACT:
        value = np.sum(np.square(inside_norm))
        derivative_outer_norm = 2.0 * inside_norm
    value += alpha * (t ** 2)

    # Forcing the steady state.
    # We add a cost for keeping X(t) and X(t + nu) the same. We use the quadratic norm for this sub-cost.
    # The larger beta and the larger nu, the closer to steady state.
    if beta > 0.0:
        for s in range(num_species):
            # Compute exp of the eigenvalues of K * (t + nu).
            wtdt = eigenvalues[s] * (t + nu)
            exp_wtdt = np.exp(wtdt)
            # Compute x_s(t) - x_s(t + nu) for that species.
            # Note that since we store V.T and U.T, we do (U.T * D * V.T).T == V * D * U
            inside_norm = xts[s] - Expm(eigenvectors_inverse[s], exp_wtdt, eigenvectors[s]).T.dot(x0s[s])
            # Increment value.
            value += beta * np.sum(np.square(inside_norm))

    # Minimize the variance of trait distribution
    if gamma > 0.0:
        # Compute the cost
        X_t = np.squeeze(np.asarray(xts)).T  # convert the list to an array
        var_Y = np.dot(np.multiply(X_t, X_t), var_Q)  # compute Var(Y)
        value += gamma * np.sum(np.square(var_Y))  # add the square of the Frobenious norm of Var(Y) to the cost

    return value


# Reshapes the array of parameters into a valid transition matrix.
# Returns 3D matrix, 3rd dimension indexes the species.
def UnflattenParameters(parameters, A, num_species):
    nstates = A.shape[0]
    # Place parameters where the adjacency matrix has value 1.
    a = A.astype(bool).flatten()
    K_all = np.zeros((nstates, nstates, num_species))
    num_nodes = A.shape[0]
    num_parameters = int(parameters.shape[0] / num_species)
    for s in range(num_species):
        matrix_parameters = np.zeros(nstates ** 2)
        matrix_parameters[a] = parameters[s * num_parameters:(s + 1) * num_parameters]
        K = np.array(matrix_parameters).reshape(num_nodes, num_nodes)
        np.fill_diagonal(K, -np.sum(K, axis=0))
        K_all[:, :, s] = K
    return K_all


# Defines bounds on transition matrix values.
def BoundParameters(num_parameters, max_rate, minimize_convergence_time,
                    f_new, x_new, f_old, x_old):
    if minimize_convergence_time:
        return np.all(x_new[:-1] <= max_rate) and np.all(x_new[:-1] >= 0.) and x_new[-1] >= 0
    return np.all(x_new <= max_rate) and np.all(x_new >= 0.)


def check_solution(params, max_rate, minimize_convergence_time):
    if minimize_convergence_time:
        return np.all(params[:-1] <= max_rate) and np.all(params[:-1] >= 0.) and params[-1] >= 0
    return np.all(params <= max_rate) and np.all(params >= 0.)


# Optimizes the rates to minimize convergence time.
# Initially this function normalizes X_init, Y_desired and max_rate so that different range of values can use the same alpha and beta.
def Optimize(Y_desired, A, X_init, Q, var_Q, max_rate, gamma=0, warm_start_parameters=None, specified_time=None,
             minimize_convergence_time=True, stabilize_robot_distribution=True, allow_trait_overflow=False,
             norm_order=2, analytical_gradients=True, verify_gradients=False, minimize_variance=False,
             max_meta_iteration=200, max_error=1e3, verbose=False):
    assert norm_order in (1, 2)
    assert (specified_time is None) == minimize_convergence_time

    global current_iteration
    current_iteration = 0

    if norm_order == 1 and allow_trait_overflow:
        mode = ABSOLUTE_AT_LEAST
    elif norm_order == 1 and not allow_trait_overflow:
        mode = ABSOLUTE_EXACT
    elif norm_order == 2 and allow_trait_overflow:
        mode = QUADRATIC_AT_LEAST
    elif norm_order == 2 and not allow_trait_overflow:
        mode = QUADRATIC_EXACT

    # Set base parameters. They should work for most input values.
    alpha = 1. if minimize_convergence_time else 0.
    if minimize_variance:
        beta = 10. if stabilize_robot_distribution else 0.
    else:
        beta = 5. if stabilize_robot_distribution else 0.
    gamma = gamma if minimize_variance else 0.
    nu = 1. if stabilize_robot_distribution else None
    margin = 0. if allow_trait_overflow else None

    # Normalize input parameters.
    sum_X = float(np.sum(X_init))
    X_init = X_init.astype(np.float) / sum_X * 800.
    Y_desired = Y_desired.astype(np.float) / sum_X * 800.
    old_max_rate = max_rate
    max_rate = 2.
    expected_convergence_time = 1.

    # Initial parameters (only where the adjacency matrix has a 1).
    num_species = X_init.shape[1]
    num_nonzero_elements = np.sum(A)
    if warm_start_parameters is None:
        init_parameters = np.random.rand(num_nonzero_elements * num_species) * max_rate
        if minimize_convergence_time:
            init_parameters = np.concatenate([
                init_parameters,
                np.array([np.random.rand() * expected_convergence_time * 2.])
            ], axis=0)
    else:
        init_parameters = warm_start_parameters

    # Bound by max rate.

    bound_fun = lambda *args, **kargs: BoundParameters(num_nonzero_elements * num_species, max_rate,
                                                       minimize_convergence_time, *args, **kargs)
    bounds = [(0., max_rate)] * num_nonzero_elements * num_species
    if minimize_convergence_time:
        bounds.append((0., None))

    if analytical_gradients:
        cost_fun = lambda x: Cost(x, Y_desired, A, X_init, Q, var_Q, alpha=alpha, specified_time=specified_time,
                                  beta=beta, gamma=gamma, nu=nu, mode=mode, margin=margin)
        minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds, 'jac': True,
                            'options': {'disp': False, 'ftol': 1e-3, 'maxiter': 100}}
    else:
        cost_fun = lambda x: cost_without_grad(x, Y_desired, A, X_init, Q, var_Q, alpha=alpha,
                                               specified_time=specified_time, beta=beta, gamma=gamma, nu=nu, mode=mode,
                                               margin=margin)
        minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds, 'jac': False,
                            'options': {'disp': False, 'ftol': 1e-3, 'maxiter': 100}}

    # Check gradient if requested.
    if verify_gradients and analytical_gradients:
        for i in range(10):
            gradient_parameters = np.random.rand(*init_parameters.shape)
            CheckGradient(lambda x: Cost(x, Y_desired, A, X_init, Q, var_Q, alpha=alpha, specified_time=specified_time,
                                         beta=beta, nu=nu, mode=mode, margin=margin), gradient_parameters)

    # Basinhopping function.
    success = False
    global meta_iteration
    meta_iteration = 0
    while not success:
        meta_iteration += 1
        if meta_iteration > max_meta_iteration:
            break
        # print('\nMeta iteration %i...' % meta_iteration)
        # It happens very rarely that the eigenvector matrix becomes close to singular and
        # cannot be inverted. In that case, we simply restart the optimization.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                ret = scipy.optimize.basinhopping(cost_fun, init_parameters, accept_test=bound_fun,
                                                  minimizer_kwargs=minimizer_kwargs, niter=50, niter_success=6,
                                                  callback=Print if verbose else None)

            constraints_satisfied = check_solution(ret.x, max_rate, minimize_convergence_time)
            success = (ret.fun < max_error) and constraints_satisfied
            if (ret.fun < 1e3) and ~constraints_satisfied:
                dummy = 1
        except (ValueError, np.linalg.linalg.LinAlgError):
            # Make completely new random elements.
            init_parameters = np.random.rand(num_nonzero_elements * num_species) * max_rate
            if minimize_convergence_time:
                init_parameters = np.concatenate([
                    init_parameters,
                    np.array([np.random.rand() * expected_convergence_time * 2.])
                ], axis=0)
            success = False

    final_parameters = np.copy(ret.x)

    # Remove the optimized t.
    if minimize_convergence_time:
        optimal_t = ret.x[-1]
        K = UnflattenParameters(ret.x[:-1], A, num_species)
    else:
        optimal_t = specified_time
        K = UnflattenParameters(ret.x, A, num_species)

    # Renormalize.
    optimal_t *= max_rate / old_max_rate
    K *= old_max_rate / max_rate
    X_init *= sum_X
    Y_desired *= sum_X

    if verbose:
        Y = simulation.ComputeY(optimal_t, K, X_init, Q)
        if allow_trait_overflow:
            error = np.sum(np.maximum(Y_desired-Y, np.zeros(Y.shape))) / np.sum(Y_desired)
        else:
            error = np.sum(np.abs(Y_desired - Y)) / (np.sum(Y_desired) * 2.)

        print('\nConverged after %i meta iterations' % meta_iteration)
        print('\nConstraints satisfied')
        print('\nTrait mismatch error (at time %.2f): %.3f%%' % (optimal_t, error * 100.))
        print('Final cost:', ret.fun)

    # Return transition matrices (3D matrix for all species)
    return K, optimal_t, final_parameters, success


if __name__ == '__main__':
    import graph
    import trait_matrix
    num_nodes = 8
    num_traits = 4
    num_species = 4
    robots_per_species = 200
    max_rate = 2.

    g = graph.Graph(num_nodes)
    X_init = g.CreateRobotDistribution(num_species, robots_per_species, site_restrict=range(0, int(num_nodes / 2)))
    Q = trait_matrix.CreateRandomQ(num_species, num_traits)
    X_final = g.CreateRobotDistribution(num_species, robots_per_species, site_restrict=range(int(num_nodes / 2),
                                                                                             num_nodes))
    Y_desired = X_final.dot(Q)
    A = g.AdjacencyMatrix()

    print('Checking gradients...')
    _, _, p = Optimize(Y_desired, A, X_init, Q, max_rate, verify_gradients=True, verbose=True)
    print('Trying warm-starting...')
    Optimize(Y_desired, A, X_init, Q, max_rate, warm_start_parameters=p, verbose=True)

    print('Trying different number of robots...')
    X_init = X_init.astype(np.float) * 10.
    Y_desired = Y_desired.astype(np.float) * 10.
    _, _, p = Optimize(Y_desired, A, X_init, Q, max_rate, verbose=True)

    print('Trying different rate...')
    max_rate = 4.
    _, _, p = Optimize(Y_desired, A, X_init, Q, max_rate, verbose=True)
