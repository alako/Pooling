from pgmpy.models import BayesianModel
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


def create_merged_dag(bn_list, merging_parents_variant=0):
    """

    :param bn_list:
    :param merging_parents_variant:
    - 0: combine
    - 1: after Feng et al.
        if node is interior 'single': copy parents from the BN that are not subset of the intersection
        if node is interior 'double': merge parents from the BNs (!) that are not subset of the intersection
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
    return decycle(weighted_graph)


def decycle(weighted_graph, greedy=True):
    cycles = list(nx.simple_cycles(weighted_graph))
    # Greedy option 1
    if cycles:
        if greedy:
            while len(cycles) != 0:
                edges = list(weighted_graph.edges.data("weight"))
                edges.sort(key=lambda x: x[2])
                min_w = edges[0][2]
                min_edges = [(u, v, w) for (u, v, w) in edges if w == min_w]
                edges_c = []
                for (u, v, w) in min_edges:
                    c = []
                    for cycle in cycles:
                        if u in cycle and v in cycle:
                            c.append(cycle)
                    edges_c.append((u, v, w, c, len(c)))
                edges_c.sort(key=lambda x: x[4], reverse=True)
                weighted_graph.remove_edge(edges_c[0][0], edges_c[0][1])
                for cycle in edges_c[0][3]:
                    cycles.remove(cycle)
            acyclic_graph = weighted_graph
        # Brute-force
        else:
            edges = list(weighted_graph.edges.data("weight"))
            edges.sort(key=lambda x: x[2])
            found_acyclic = False
            acyclic_graphs = []
            all_combinations = []
            for edge in edges:
                new_graph = weighted_graph.copy()
                new_graph.remove_edge(edge)
                found_acyclic = nx.is_directed_acyclic_graph(new_graph)
                if found_acyclic:
                    acyclic_graphs.append(new_graph)
                all_combinations.append([edge])

            while not found_acyclic:
                new_all_combinations = []
                for comb in all_combinations:
                    for edge in edges:
                        new_comb = comb.copy()
                        new_comb.append(edge)
                        new_all_combinations.append(new_comb)
                        # check this comb
                        new_graph = weighted_graph.copy()
                        new_graph.remove_edges(new_comb)
                        found_acyclic = nx.is_directed_acyclic_graph(new_graph)
                        if found_acyclic:
                            acyclic_graphs.append(new_graph)
                # select the best
                all_combinations = new_all_combinations
            acyclic_graph = acyclic_graphs[0]
    else:
        acyclic_graph = weighted_graph
    graph = list(acyclic_graph.edges)
    graph_dict = {}
    for node in list(acyclic_graph.nodes):
        graph_dict[node] = set()
    for (e, v) in graph:
        graph_dict[v].add(e)
    return graph, graph_dict


