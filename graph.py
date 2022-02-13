import networkx as nx
import numpy as np

import nxmod


class Graph(object):
    def __init__(self, M, fully_connected=False):
        # M: Number of nodes.
        self._num_nodes = M
        if fully_connected:
            self._graph = nx.complete_graph(M)
        else:
            self._graph = nx.connected_watts_strogatz_graph(M, 3, 0.6)

    def AdjacencyMatrix(self):
        return np.squeeze(np.asarray(nx.to_numpy_matrix(self._graph))).astype(np.int32)

    def Plot(self, Y, ax=None):
        # Y: MxU matrix.
        # Returns: fig
        return nxmod.DrawCircular(Y, self._graph, width=3, ax=ax)


    def CreateRobotDistribution(self, num_species, num_robot_per_species, site_restrict=None):
        if site_restrict is None:
            site_restrict = range(self._num_nodes)
        X = np.zeros(((self._num_nodes, num_species)))
        for s in range(num_species):
            R = np.random.choice(site_restrict, size=num_robot_per_species)
            for m in site_restrict:
                X[m, s] = np.sum(R == m)
        return X.astype(np.int32)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import continuous_trait_matrix as trait_matrix

    num_nodes = 8
    num_traits = 6
    num_species = 6
    num_robots_per_species = 200

    graph = Graph(num_nodes)
    X = graph.CreateRobotDistribution(num_species, num_robots_per_species)
    Q = trait_matrix.CreateRandomQ(num_species, num_traits)
    graph.Plot(X.dot(Q))

    plt.show()
