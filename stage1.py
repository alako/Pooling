from pgmpy.models import BayesianModel
from more_itertools import powerset
from random import randrange

import networkx as nx
"""
The first stage includes: 
- creating a combined graph (acyclicity not guaranteed) based on the list of BNs,
- reducing the number of edges to achieve a DAG 
"""


def bn_nodes_intersection(bn_list):
    common_variables = set(bn_list[0].nodes)
    for bn in bn_list:
        nodes = set(bn.nodes)
        common_variables = common_variables.intersection(nodes)
    return common_variables


def bn_nodes_union(bn_list):
    common_variables = set()
    for bn in bn_list:
        nodes = set(bn.nodes)
        common_variables = common_variables.union(nodes)
    return common_variables


def is_interior(z, bn_list):
    """

    :param z:
    :param bn_list:
    :return:
        if z is interior: a list of BNs for which parents are not subset of intersection of all nodes,
        if not: empty list [works as False]
    """
    interior = False
    bns = []
    for bn in bn_list:
        if bn.has_node(z):
            parents = bn.get_parents(z)
            if set(parents) <= set(bn_nodes_intersection(bn_list)):
                interior = True
                bns.append(bn)
    return bns


def is_interior_single(z, bn_list):
    """

    :param z:
    :param bn_list:
    :return:
        if z is interior single: a BN for which parents are not subset of intersection of all nodes,
        if not: None [works as False]
    """
    interior_single = False
    count = 0
    single_bn = None
    for bn in bn_list:
        if bn.has_node(z):
            parents = bn.get_parents(z)
            if set(parents) <= set(bn_nodes_intersection(bn_list)):
                interior_single = True
            if not set(parents) <= set(bn_nodes_intersection(bn_list)):
                count += 1
                single_bn = bn
    if count != 1:
        interior_single = False
    if not interior_single:
        single_bn = None
    return single_bn


def is_exterior(z, bn_list):
    exterior = True
    for bn in bn_list:
        if bn.has_node(z):
            parents = bn.get_parents(z)
            if not set(parents) <= set(bn_nodes_intersection(bn_list)):
                exterior = False
    return exterior


def combine_parents(z, bn_list):
    combined_parents = set()
    for bn in bn_list:
        if bn.has_node(z):
            parents = bn.get_parents(z)
            combined_parents = combined_parents.union(parents)
    return combined_parents


def is_unique(z, bn_list):
    z_occurrence = 0
    for bn in bn_list:
        if bn.has_node(z):
            z_occurrence += 1
    if z_occurrence == 1:
        return True,
    return False


def find_bn_with_z(z, bn_list):
    for bn in bn_list:
        if bn.has_node(z):
            return bn
    return None


def find_all_bns_with_z(z, bn_list):
    bns_with_z = []
    for bn in bn_list:
        if bn.has_node(z):
            bns_with_z.append(bn)
    return bn_list


def merge_parents(node, bn_list, variant=0):
    """
    Combine parents from all BNs
    :param node:
    :param bn_list:
    :param variant:
    - 0: combine
    - 1: after Feng et al.
        if node is interior 'single': copy parents from the BN that are not subset of the intersection
        if node is interior 'double': merge parents from the BNs (!) that are not subset of the intersection
    :return: set of all parents assigned to the node
    """
    # if node belongs to only one BN
    if is_unique(node, bn_list):
        # copy parents from that unique BN
        bn = find_bn_with_z(node, bn_list)
        node_new_parents = bn.get_parents(node)
    elif variant == 1:
        # find bns where parents are not subset of the intersection
        bns = is_interior(node, bn_list)
        if len(bns) == 1:
            # copy parents from the BN that are not subset of the intersection
            node_new_parents = bns[0].get_parents(node)
        elif len(bns) > 1:
            # merge parents from the BNs (!) that are not subset of the intersection
            node_new_parents = combine_parents(node, bns)
        else:
            # node is exterior, so combine parents
            node_new_parents = combine_parents(node, bn_list)
    else:
        # variant 0
        node_new_parents = combine_parents(node, bn_list)
    return node_new_parents


def create_merged_dag(bn_list, merging_parents_variant=0, greedy=2):
    """

    :param bn_list:
    :param merging_parents_variant:
    - 0: combine
    - 1: after Feng et al.
        if node is interior 'single': copy parents from the BN that are not subset of the intersection
        if node is interior 'double': merge parents from the BNs (!) that are not subset of the intersection
    :param greedy:
    - 0: decycling fully brute-force
    - 1: decycling half brute-force, half-greedy
    - 2: decycling fully greedy
    :return:
    """
    weighted_graph = nx.DiGraph()
    all_nodes = bn_nodes_union(bn_list)
    for node in all_nodes:
        # find new parents
        node_new_parents = merge_parents(node, bn_list, merging_parents_variant)
        # save edges with weights corresponding to the # of occurrences
        for parent in node_new_parents:
            w = 0
            for bn in bn_list:
                if bn.has_edge(parent, node):
                    w += 1
            weighted_graph.add_edge(parent, node, weight=w)
    return decycle(weighted_graph, greedy)


def find_cycles_containing_edges(edges, cycles):
    edges_c = []
    for (u, v, w) in edges:
        c = []
        for cycle in cycles:
            if u in cycle and v in cycle:
                c.append(cycle)
        edges_c.append((u, v, w, c, len(c)))
    return edges_c


def greedy_decycling(weighted_graph, cycles):
    while len(cycles) != 0:
        edges = list(weighted_graph.edges.data("weight"))
        # sort edges in decreasing order using their weights
        edges.sort(key=lambda x: x[2])
        min_w = edges[0][2]
        # consider the subset of edges with the smallest weight
        min_edges = [(u, v, w) for (u, v, w) in edges if w == min_w]
        edges_c = find_cycles_containing_edges(min_edges, cycles)
        # sort edges by the number of cycles they belong to (from max to min)
        edges_c.sort(key=lambda x: x[4], reverse=True)
        max_c_edges = [(u, v, w, c, lc) for (u, v, w, c, lc) in edges_c if lc == edges_c[0][4]]
        i = randrange(len(max_c_edges))
        weighted_graph.remove_edge(edges_c[i][0], edges_c[i][1])
        for cycle in edges_c[0][3]:
            cycles.remove(cycle)
    return weighted_graph


def filter_edges_in_cycles(edges, cycles):
    edges_in_cycles = []
    for (u, v, w) in edges:
        in_cycle = False
        for cycle in cycles:
            if u in cycle and v in cycle:
                in_cycle = True
        if in_cycle:
            edges_in_cycles.append((u, v, w))
    return edges_in_cycles


def half_greedy_decycling(weighted_graph, cycles):
    while len(cycles) != 0:
        edges = list(weighted_graph.edges.data("weight"))
        edges_in_cycles = filter_edges_in_cycles(edges, cycles)
        # sort edges in decreasing order using their weights
        edges_in_cycles.sort(key=lambda x: x[2])
        min_w = edges_in_cycles[0][2]
        # consider the subset of edges with the smallest weight
        min_edges = [(u, v, w) for (u, v, w) in edges_in_cycles if w == min_w]
        edges_c = find_cycles_containing_edges(min_edges, cycles)
        # check if deleting all min_edges results in an acyclic graph
        new_graph = weighted_graph.copy()
        min_weight = 0
        for edge in min_edges:
            min_weight += edge[2]
            new_graph.remove_edge(edge)
        if nx.is_directed_acyclic_graph(new_graph):
            # if yes, find the smallest subset that gives us an acyclic graph
            dag = new_graph
            subsets = powerset(edges_c)
            for subset in subsets:
                new_graph = weighted_graph.copy()
                for edge in subset:
                    new_graph.remove_edge(edge)
                if nx.is_directed_acyclic_graph(new_graph):
                    return new_graph
            return dag
        else:
            # if no, delete edges that reduce the number of cycles and go back
            for edge in edges_c:
                remove = False
                for cycle in edge[3]:
                    if cycle in cycles:
                        remove = True
                if remove:
                    weighted_graph.remove_edge(edge[0], edge[1])
                    for cycle in edge[3]:
                        cycles.remove(cycle)
    return weighted_graph


def brute_force_decycling(weighted_graph, cycles):
    edges = list(weighted_graph.edges.data("weight"))
    edges_in_cycles = filter_edges_in_cycles(edges, cycles)
    edges_c = find_cycles_containing_edges(edges_in_cycles, cycles)
    min_weight = 0
    # TODO: weight corresponding to confirmation scores
    for (u, v, w) in edges_in_cycles:
        min_weight += w
    subsets = powerset(edges_c)
    # TODO: sort subsets using cumulative weight
    for subset in subsets:
        weight = 0
        for (u, v, w) in subset:
            weight += w
        if weight <= min_weight:
            new_graph = weighted_graph.copy()
            for edge in subset:
                new_graph.remove_edge(edge)
            if nx.is_directed_acyclic_graph(new_graph):
                dag = new_graph
                min_weight = weight
    return dag


def decycle(weighted_graph, greedy=1):
    cycles = list(nx.simple_cycles(weighted_graph))
    if cycles:
        if greedy == 2:
            acyclic_graph = greedy_decycling(weighted_graph, cycles)
        elif greedy == 1:
            acyclic_graph = half_greedy_decycling(weighted_graph, cycles)
        else:
            acyclic_graph = brute_force_decycling(weighted_graph, cycles)
    else:
        acyclic_graph = weighted_graph
    graph = list(acyclic_graph.edges)
    graph_dict = {}
    for node in list(acyclic_graph.nodes):
        graph_dict[node] = set()
    for (e, v) in graph:
        graph_dict[v].add(e)
    return graph, graph_dict


