from pgmpy.models import BayesianModel
import networkx as nx


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
    interior = False
    for bn in bn_list:
        if bn.has_node(z):
            parents = bn.get_parents(z)
            if set(parents) <= set(bn_nodes_intersection(bn_list)):
                interior = True
    return interior


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


def pool_graph(bn_list):
    graph = []
    graph_dict = {}
    weighted_graph = nx.DiGraph()
    # if z exists in only one BN
    all_nodes = bn_nodes_union(bn_list)
    for node in all_nodes:
        if is_unique(node, bn_list):
            # take parents from the unique BN that contains node
            for bn in bn_list:
                if bn.has_node(node):
                    node_new_parents = bn.get_parents(node)
        else:
            # combine parents
            # TODO: some other options
            node_new_parents = combine_parents(node, bn_list)
        for parent in node_new_parents:
            graph.append((parent, node))
            w = 0
            for bn in bn_list:
                if bn.has_edge(parent, node):
                    w += 1
            weighted_graph.add_edge(parent, node, weight=w)
        graph_dict[node] = node_new_parents
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


