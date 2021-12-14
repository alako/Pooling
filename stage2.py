from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np
from stage1 import *


def transform_cpd_to_table(cpd):
    if cpd.ndim == 1:
        col = []
        for x in cpd:
            col.append([x])
        return col
    table = []
    for x in cpd:  # node states
        parents_list = []
        for y in x:  # parent states
            parents_list.append(cpd[x, y])
        table.append(parents_list)
    return table


def parent_states_combinations(parents, parents_card, bn_list):
    all_combinations = []
    if not parents:
        return [[]]
    from functools import reduce
    combinations_no = reduce(lambda x, y: x * y, parents_card)  # multiply cardinality of all parents
    for parent in parents:
        bn = find_bn_with_z(parent, bn_list)
        parent_states = bn.get_cpds(parent).state_names[parent]
        if len(all_combinations) == 0:
            for state in parent_states:
                all_combinations.append([state])
        else:
            while len(all_combinations) < combinations_no:
                new_all_combinations = []
                for comb in all_combinations:
                    for state in parent_states:
                        new_comb = comb.copy()
                        new_comb.append(state)
                        new_all_combinations.append(new_comb)
                all_combinations = new_all_combinations
    return all_combinations


def normalize_cpd(values):
    ncol = len(values[0])
    normalized_values = []
    for c in range(ncol):
        col = values[:, c]
        sum_col = sum(col)
        if sum_col != 1:
            normalized_col = []
            for x in col:
                x = x/sum_col
                normalized_col.append(x)
            normalized_values.append(normalized_col)
        else:
            normalized_values.append(col)
    normalized_values = np.array(normalized_values)
    normalized_values = np.transpose(normalized_values)
    return normalized_values


def combine_cpd(z, card, parents, parents_card, bn_list):
    # for nodes contained by only one BN
    if is_unique(z, bn_list):
        bn = find_bn_with_z(z, bn_list)
        values = transform_cpd_to_table(bn.get_cpds(z).values)
        parents = bn.get_parents(z)
        parents_card = []
        for parent in parents:
            parents_card.append(bn.get_cardinality(parent))
    else:
        bn_list = find_all_bns_with_z(z, bn_list)
        all_parents_combinations = parent_states_combinations(parents, parents_card, bn_list)
        values = []
        # TODO: extend for more BNs
        bn1 = bn_list[0]
        bn2 = bn_list[1]
        infer1 = VariableElimination(bn1)
        infer2 = VariableElimination(bn2)
        parents1 = bn1.get_parents(z)
        parents2 = bn2.get_parents(z)
        for z_state in range(card):
            parent_values = []
            for parent_comb in all_parents_combinations:
                evidence = {}
                for idx, state in enumerate(parent_comb):
                    evidence[parents[idx]] = state
                # OPTION 1: Bradley variant of linear pooling
                pooling_possible = True
                for parent in parents:
                    if (not bn1.has_node(parent)) or (not bn2.has_node(parent)):
                        pooling_possible = False
                if pooling_possible:
                    posterior_z1 = infer1.query([z], evidence=evidence).values[z_state]
                    posterior_z2 = infer2.query([z], evidence=evidence).values[z_state]
                    posterior_z = (posterior_z1+posterior_z2)/2
                else:
                    # TODO: consider the case when a node is parentless on other BNs
                    # OPTION 2: Feng variant of combining CPDs
                    evidence1, evidence2 = {}, {}
                    for p in parents1:
                        evidence1[p] = evidence[p]
                    for p in parents2:
                        evidence2[p] = evidence[p]
                    posterior_z1 = infer1.query([z], evidence=evidence1).values[z_state]
                    posterior_z2 = infer2.query([z], evidence=evidence2).values[z_state]
                    posterior_z = (posterior_z1 + posterior_z2) - (posterior_z1 * posterior_z2)
                parent_values.append(posterior_z)
            # TODO: no parent case, prior combine
            values.append(parent_values)
    values = normalize_cpd(np.array(values))
    cpd = TabularCPD(z, card, values, evidence=parents, evidence_card=parents_card)
    return cpd


def add_cpds(dag, dag_dict, bn_list):
    # Defining the network structure
    pooled_bn = BayesianModel(dag)
    for node in dag_dict:
        parents = list(dag_dict[node])
        card = 2
        for bn in bn_list:
            if bn.has_node(node):
                card = bn.get_cardinality(node)
        parents_card = []
        for parent in parents:
            for bn in bn_list:
                if bn.has_node(parent):
                    parent_card = bn.get_cardinality(parent)
            parents_card.append(parent_card)
        cpd = combine_cpd(node, card, parents, parents_card, bn_list)
        pooled_bn.add_cpds(cpd)
    print(pooled_bn.check_model())
    return pooled_bn
