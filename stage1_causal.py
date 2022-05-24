from typing import List

import networkx as nx
from pgmpy.independencies import Independencies
from pgmpy.estimators import PC
import random

from pgmpy.models import BayesianNetwork

from stage1 import decycle


def find_common_independencies(bn_list):
    common_independencies = set(bn_list[0].get_independencies().closure().independencies)
    for bn in bn_list:
        ind = set(bn.get_independencies().closure().independencies)
        common_independencies.intersection_update(ind)
    common_independencies = list(common_independencies)
    print(common_independencies)
    return Independencies(*common_independencies)


def undirected_edges_by_occurrence(pdag, bn_list):
    """
    :return: a list of undirected edges (u,v,w) with corresponding weight w=|# right - # left|
    the list contains only one direction of each edge -> the one with higher w
    """
    undirected_edges = []
    visited_edges = set()
    for (u, v) in pdag.undirected_edges:
        if (v, u) not in visited_edges:
            a, b = 0, 0
            for bn in bn_list:
                if bn.has_edge(u, v):
                    a += 1
                elif bn.has_edge(v, u):
                    b += 1
            visited_edges.add((u, v))
            if a > b:
                parent, node = u, v
            elif b > a:
                parent, node = v, u
            else:
                if random.choice([True, False]):
                    parent, node = u, v
                else:
                    parent, node = v, u
            undirected_edges.append((parent, node, abs(a - b)))
    return undirected_edges


def propagate_orientation(tree, dag, node):
    neighbors = tree[node].copy()
    if not neighbors:
        return dag
    for n in neighbors:
        dag.add_edge(node, n)
    for n in neighbors:
        tree.remove_edge(n, node)
    for n in neighbors:
        dag = propagate_orientation(tree, dag, n)
    return dag


def find_dags_from_tree(tree: nx.Graph):
    all_dags = []
    # pick the root of the tree
    for node in tree:
        neighbors = tree[node]
        # to avoid adding v-structures there are two ways to set directions of root edges
        # 1st case: 0 incoming edges
        dag = nx.DiGraph()
        dag = propagate_orientation(tree.copy(), dag, node)
        all_dags.append(dag)

        # 2nd case: 1 incoming edge
        for n in neighbors:
            dag = nx.DiGraph()
            # adding incoming edge
            dag.add_edge(n, node)
            other_neighbors = neighbors.copy()
            other_neighbors.pop(n, 'None')
            for o in other_neighbors:
                dag = propagate_orientation(tree.copy(), dag, o)
            all_dags.append(dag)
    return all_dags


def agreement_score(dag: nx.DiGraph, bn_list: List[BayesianNetwork]):
    edge_scores = []
    for (u, v) in dag.edges:
        yes, no = 0, 0
        for bn in bn_list:
            if bn.has_edge(u, v):
                yes += 1
            elif bn.has_edge(v, u):
                no += 1
        edge_scores.append(yes/(yes+no))
    return sum(edge_scores)/len(edge_scores)


def best_trees_orientations(trees: List[nx.Graph], bn_list):
    best_dags = []
    for tree in trees:
        all_dags = find_dags_from_tree(tree)
        scores = [(agreement_score(dag, bn_list), dag) for dag in all_dags]
        best_dag = max(scores, key=lambda item: item[0])[1]
        best_dags.append(best_dag)
    return best_dags


def pdag_to_dag(pdag, bn_list):
    """
    Greedy approach to obtain a DAG from a PDAG given edge directions in bn_list
    (PDAG is assumed to be based on bn_list)
    We cannot add new v-structures or introduce cycles
    weight = |# right - # left|
    """

    directed_edges = [e for e in pdag.directed_edges]
    dag = nx.DiGraph(directed_edges)

    # returns undirected edges weighted by occurrences of directions
    undirected_edges = undirected_edges_by_occurrence(pdag, bn_list)
    undirected_graph = nx.Graph()
    undirected_graph.add_weighted_edges_from(undirected_edges)

    # find connected components of undirected edges
    trees = [undirected_graph.subgraph(c).copy() for c in nx.connected_components(undirected_graph)]

    # there are several possible orientations of a connected component (tree) that do not introduce v-structures
    # pick the one that is more common in bn_list
    best_small_dags = best_trees_orientations(trees, bn_list)

    # add all edges together to form a final dag
    for b in best_small_dags:
        dag = nx.compose(dag, b)

    return dag.edges, nx.to_dict_of_lists(dag)



    # sort edges in decreasing order using their weights
    # undirected_edges.sort(key=lambda x: x[2], reverse=True)

    # for (u, v, w) in undirected_edges:
    #     graph.add_edge(u, v)
    #     pdag.remove_edge(v, u)
    #
    #
    #     # Check if this edge creates a cycle
    #     # cycles = list(nx.simple_cycles(graph))
    #     # if cycles:
    #     #     graph.remove_edge(u, v)
    #     #     # This could add a v-structure if undirected edges form a cycle
    #     #     u, v = v, u
    #     #     graph.add_edge(u, v)
    #
    #     # propagate direction to avoid new V-structures
    #     parents = pdag.pred[v]
    #     remove_edges = set()
    #     for p in parents:
    #         if pdag.has_edge(v, p):
    #             remove_edges.add((p, v))
    #     for (a, b) in remove_edges:
    #         pdag.remove_edge(a, b)

    # Prepare dictionary
    # graph_dict = {}
    # for v in list(pdag.nodes):
    #     graph_dict[v] = set()
    # for (e, v) in pdag.edges:
    #     graph_dict[v].add(e)
    # return list(pdag.edges), graph_dict


def independencies_to_pdag(ind):
    pass


def create_merged_dag(bn_list):
    """

    :param bn_list:
    :return:
    """
    ind = find_common_independencies(bn_list)
    pdag = PC(independencies=ind).estimate(ci_test="independence_match", return_type="pdag")
    # pdag = independencies_to_pdag(ind)
    dag, dag_dict = pdag_to_dag(pdag, bn_list)
    return pdag, dag, dag_dict
