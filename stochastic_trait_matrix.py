import numpy as np
import sys

import utils


def CreateRandomQ(S, U, num_non_cumu_traits=0):
    # Initial all the elements of the matrix to random values
    A = np.ones([int(S/2), U]) * 2
    B = np.ones([(S - int(S/2)), U]) * 10
    C = np.concatenate((A, B), axis=0)
    Q = np.random.rand(S, U) * C

    # Binary representation for non-cumulative traits
    for k in range(num_non_cumu_traits):
        Q.T[k] = np.random.randint(2, size=S)

    # make the matrix a little sparse.
    num = np.random.randint(U/2, (U - num_non_cumu_traits) * S + 1)
    for n in range(num):
      i = np.random.randint(0, S)
      j = np.random.randint(num_non_cumu_traits, U)
      Q[i, j] = 0
    return Q.astype(float)


def CreateRankedQ(S, U, num_non_cumu_traits=0):
    # Guarantees that Q has maximum rank (== U).
    assert U <= S
    Q = CreateRandomQ(S, U, num_non_cumu_traits)
    while np.linalg.matrix_rank(Q) != U:
      Q = CreateRandomQ(S, U, num_non_cumu_traits)

    # Define the variance of each element of Q
    var_Q = np.random.rand(Q.shape[0], Q.shape[1])

    return Q, var_Q


if __name__ == '__main__':
    num_species = 20
    num_traits = 10
    sys.stdout.write('Generating random matrix with maximum ranks...\t')
    sys.stdout.flush()
    for i in range(num_traits):
        Q = CreateRankedQ(num_species, num_traits)
        assert np.linalg.matrix_rank(Q[0]) == num_traits
    sys.stdout.write(utils.Highlight('[DONE]\n', utils.GREEN, bold=True))
