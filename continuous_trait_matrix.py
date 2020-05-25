import numpy as np
import sys

import utils


def CreateRandomQ(S, U):
    # Initial all the elements of the matrix to random values
    Q =  np.random.rand(S,U)*10

    num = np.random.randint(0, U * S + 1)
    # make the matrix a little sparse.
    for n in range(num):
      i = np.random.randint(0, S)
      j = np.random.randint(0, U)
      Q[i, j] = 0
    return Q.astype(np.float)


def CreateRankedQ(S, U):
    # Guarantees that Q has maximum rank (== U).
    assert U <= S
    Q = CreateRandomQ(S, U)
    while np.linalg.matrix_rank(Q) != U:
      Q = CreateRandomQ(S, U)
    return Q


if __name__ == '__main__':
    num_species = 20
    num_traits = 10
    sys.stdout.write('Generating random matrix with maximum ranks...\t')
    sys.stdout.flush()
    for i in range(num_traits):
        Q = CreateRankedQ(num_species, num_traits)
        assert np.linalg.matrix_rank(Q) == num_traits
    sys.stdout.write(utils.Highlight('[DONE]\n', utils.GREEN, bold=True))
