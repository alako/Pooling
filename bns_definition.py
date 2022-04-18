import networkx as nx
from pgmpy.base import DAG
from pgmpy.models import BayesianModel, BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import numpy as np
import string


def random_BN(nodes=5, edge_prob=None):
    model = get_random(n_nodes=nodes, edge_prob=edge_prob, n_states=2)
    return model


# BN 1
def bn1():
    # Defining the network structure
    model1 = BayesianModel([('A', 'C'), ('C', 'D'), ('C', 'E')])
    # Defining the CPDs:
    cpd_a = TabularCPD('A', 2, [[0.5], [0.5]])
    cpd_c = TabularCPD('C', 2, [[0.9, 0.7],
                                [0.1, 0.3]],
                       evidence=['A'], evidence_card=[2])
    cpd_d = TabularCPD('D', 2, [[0.4, 0.7],
                                [0.6, 0.3]],
                       evidence=['C'], evidence_card=[2])
    cpd_e = TabularCPD('E', 2, [[0.5, 0.2],
                                [0.5, 0.8]],
                       evidence=['C'], evidence_card=[2])
    # Associating the CPDs with the network structure.
    model1.add_cpds(cpd_a, cpd_c, cpd_d, cpd_e)
    # Some other methods
    # print(model1.get_cpds('A'))
    model1.check_model()
    # nx.draw(model1, with_labels=True)
    # plt.show()
    return model1


# BN 2
def bn2():
    # Defining the network structure
    model2 = BayesianNetwork([('D', 'E'), ('D', 'A'), ('C', 'E')])
    # Defining the CPDs:
    # cpd_b = TabularCPD('A', 2, [[0.5, 0.5]])
    cpd_b = TabularCPD('A', 2, [[0.4, 0.7],
                                [0.6, 0.3]],
                       evidence=['D'], evidence_card=[2])
    cpd_c = TabularCPD('C', 2, [[0.3], [0.7]])
    # cpd_d = TabularCPD('D', 2, [[0.4, 0.7],
    #                             [0.6, 0.3]],
    #                    evidence=['A'], evidence_card=[2])
    cpd_d = TabularCPD('D', 2, [[0.4], [0.6]])
    cpd_e = TabularCPD('E', 2, [[0.5, 0.2, 0.4, 0.8],
                                [0.5, 0.8, 0.6, 0.2]],
                       evidence=['C', 'D'], evidence_card=[2, 2])
    # Associating the CPDs with the network structure.
    model2.add_cpds(cpd_b)
    model2.add_cpds(cpd_c, cpd_d, cpd_e)
    # Some other methods
    print(model2.get_cpds('D'))
    model2.check_model()

    # nx.draw(model2, with_labels=True)
    # plt.show()
    return model2


def bn3():
    # Defining the network structure
    model1 = BayesianModel([('A', 'C'), ('A', 'B'), ('D', 'C')])
    # Defining the CPDs:
    cpd_a = TabularCPD('A', 2, [[0.5], [0.5]])
    cpd_c = TabularCPD('C', 2, [[0.5, 0.2, 0.4, 0.8],
                                [0.5, 0.8, 0.6, 0.2]],
                       evidence=['A', 'D'], evidence_card=[2, 2])
    cpd_d = TabularCPD('D', 2, [[0.4], [0.6]])
    cpd_b = TabularCPD('B', 2, [[0.50000, 0.2],
                                [0.50000, 0.8]],
                       evidence=['A'], evidence_card=[2])
    # Associating the CPDs with the network structure.
    model1.add_cpds(cpd_a, cpd_c, cpd_d, cpd_b)
    # Some other methods
    # print(model1.get_cpds('A'))
    model1.check_model()
    # nx.draw(model1, with_labels=True)
    # plt.show()
    return model1


def bn4():
    # Defining the network structure
    model1 = BayesianModel([('A', 'C'), ('B', 'D'), ('D', 'C')])
    # Defining the CPDs:
    cpd_a = TabularCPD('A', 2, [[0.5], [0.5]])
    cpd_c = TabularCPD('C', 2, [[0.5, 0.2, 0.4, 0.8],
                                [0.5, 0.8, 0.6, 0.2]],
                       evidence=['A', 'D'], evidence_card=[2, 2])
    cpd_d = TabularCPD('D', 2, [[0.5, 0.2],
                                [0.5, 0.8]],
                       evidence=['B'], evidence_card=[2])
    cpd_b = TabularCPD('B', 2, [[0.8], [0.2]])
    # Associating the CPDs with the network structure.
    model1.add_cpds(cpd_a, cpd_c, cpd_d, cpd_b)
    # Some other methods
    # print(model1.get_cpds('A'))
    model1.check_model()
    # nx.draw(model1, with_labels=True)
    # plt.show()
    return model1


def get_random(n_nodes=5, edge_prob=0.5, n_states=None, latents=False):
    """
        Returns a randomly generated bayesian network on `n_nodes` variables
        with edge probabiliy of `edge_prob` between variables.

        Parameters
        ----------
        n_nodes: int
            The number of nodes in the randomly generated DAG.

        edge_prob: float
            The probability of edge between any two nodes in the topologically
            sorted DAG.

        n_states: int or list (array-like) (default: None)
            The number of states of each variable. When None randomly
            generates the number of states.

        latents: bool (default: False)
            If True, also creates latent variables.

        Returns
        -------
        Random DAG: pgmpy.base.DAG
            The randomly generated DAG.
       """
    if n_states is None:
        n_states = np.random.randint(low=1, high=5, size=n_nodes)
    elif isinstance(n_states, int):
        n_states = np.array([n_states] * n_nodes)
    else:
        n_states = np.array(n_states)

    if n_nodes > 26:
        n_states_dict = {i: n_states[i] for i in range(n_nodes)}
    else:
        n_states_dict = {j: n_states[i] for i, j in enumerate(list(string.ascii_uppercase[:n_nodes]))}

    dag = get_random_dag(n_nodes=n_nodes, edge_prob=edge_prob, latents=latents)
    bn_model = BayesianNetwork(dag.edges(), latents=dag.latents)
    bn_model.add_nodes_from(dag.nodes())

    cpds = []
    for node in bn_model.nodes():
        parents = list(bn_model.predecessors(node))
        cpds.append(
            TabularCPD.get_random(
                variable=node, evidence=parents, cardinality=n_states_dict
            )
        )

    bn_model.add_cpds(*cpds)
    return bn_model


def get_random_dag(n_nodes=5, edge_prob=0.5, latents=False):
    """
        Returns a randomly generated DAG with `n_nodes` number of nodes with
        edge probability being `edge_prob`.

        Parameters
        ----------
        n_nodes: int
            The number of nodes in the randomly generated DAG.

        edge_prob: float
            The probability of edge between any two nodes in the topologically
            sorted DAG.

        latents: bool (default: False)
            If True, includes latent variables in the generated DAG.

        Returns
        -------
        pgmpy.base.DAG instance: The randomly generated DAG.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> random_dag = DAG.get_random(n_nodes=10, edge_prob=0.3)
        >>> random_dag.nodes()
        NodeView((0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
        >>> random_dag.edges()
        OutEdgeView([(0, 6), (1, 6), (1, 7), (7, 9), (2, 5), (2, 7), (2, 8), (5, 9), (3, 7)])
        """
    # Step 1: Generate a matrix of 0 and 1. Prob of choosing 1 = edge_prob
    adj_mat = np.random.choice(
        [0, 1], size=(n_nodes, n_nodes), p=[1 - edge_prob, edge_prob]
    )

    # Step 2: Use the upper triangular part of the matrix as adjacency.
    edges = nx.convert_matrix.from_numpy_matrix(
        np.triu(adj_mat, k=1), create_using=nx.DiGraph
    ).edges()

    if n_nodes > 26:
        nodes = list(range(n_nodes))
    else:
        nodes = list(string.ascii_uppercase[:n_nodes])
        d = {i: x for i, x in enumerate(list(string.ascii_uppercase))}
        letter_edges = []
        for (u, v) in edges:
            letter_edges.append((d[u], d[v]))
        edges = letter_edges
    dag = DAG(edges)
    dag.add_nodes_from(nodes)
    if latents:
        dag.latents = set(
            np.random.choice(
                dag.nodes(), np.random.randint(low=0, high=len(dag.nodes()))
            )
        )
    return dag
