from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np
from pgmpy.models import BayesianNetwork
from stage1 import *
from functools import reduce
"""
The second stage includes:
- computing merged conditional probability distributions (CPD)
- adding them to a DAG
"""


def transform_cpd_to_table(cpd):
    if cpd.ndim == 1:
        col = []
        for x in cpd:
            col.append([x])
        return col
    return cpd
    # table = []
    # for x in cpd:  # node states
    #     parents_list = []
    #     for y in x:  # parent states
    #         parents_list.append(cpd[x, y])
    #     table.append(parents_list)
    # return table


def parent_states_combinations(parents, parents_card, bn_list):
    all_combinations = []
    if not parents:
        return [[]]
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


def is_bradley_pooling_possible(parents, bn_list):
    bradley_pooling_possible = True
    for parent in parents:
        for bn in bn_list:
            if not bn.has_node(parent):
                bradley_pooling_possible = False
    return bradley_pooling_possible


def bradley_pooling(infer, z, evidence, z_state):
    posterior_z = 0
    for inf in infer:
        posterior_zi = inf.query([z], evidence=evidence).values[z_state]
        posterior_z += posterior_zi
    posterior_z /= len(infer)
    return posterior_z


def feng_pooling(infer, z, evidence, z_state, parents_all):
    evidence_all = []
    for parents_i in parents_all:
        for p in parents_i:
            evidence_i = {p: evidence[p]}
            evidence_all.append(evidence_i)
    x, y = 0, 1
    for inf, evidence_i in zip(infer, evidence_all):
        posterior_zi = inf.query([z], evidence=evidence_i).values[z_state]
        x += posterior_zi
        y *= posterior_zi
    return x - y


def combine_cpd(z, card, parents, parents_card, bn_list, variant='bradley'):
    # for nodes contained by only one BN
    if is_unique(z, bn_list):
        bn = find_bn_with_z(z, bn_list)
        # values = bn.get_cpds(z).values
        values = transform_cpd_to_table(bn.get_cpds(z).values)
    else:
        bn_list = find_all_bns_with_z(z, bn_list)
        all_parents_combinations = parent_states_combinations(parents, parents_card, bn_list)
        values = []
        infer = [VariableElimination(bn) for bn in bn_list]
        parents_all = [bn.get_parents(z) for bn in bn_list]
        for z_state in range(card):
            parent_values = []
            for parent_comb in all_parents_combinations:
                evidence = {}
                for idx, state in enumerate(parent_comb):
                    evidence[parents[idx]] = state
                if variant == 'bradley' and is_bradley_pooling_possible(parents, bn_list):
                    # OPTION 1: Bradley variant of linear pooling
                    posterior_z = bradley_pooling(infer, z, evidence, z_state)
                else:
                    # OPTION 2: Feng variant of combining CPDs
                    posterior_z = feng_pooling(infer, z, evidence, z_state, parents_all)
                parent_values.append(posterior_z)
            values.append(parent_values)
    values = normalize_cpd(np.array(values))
    cpd = TabularCPD(z, card, values, evidence=parents, evidence_card=parents_card)
    return cpd


def add_merged_cpds(dag, dag_dict, bn_list, cpd_variant='bradley'):
    # Defining the network structure
    pooled_bn = BayesianNetwork(dag)
    all_nodes = bn_nodes_union(bn_list)
    for node in all_nodes:
        if not pooled_bn.has_node(node):
            pooled_bn.add_node(node)
            dag_dict[node] = set()
    for node in dag_dict:
        parents = list(dag_dict[node])
        card = 2
        for bn in bn_list:
            if bn.has_node(node):
                card = bn.get_cardinality()[node]
        parents_card = []
        for parent in parents:
            for bn in bn_list:
                if bn.has_node(parent):
                    parent_card = bn.get_cardinality()[parent]
            parents_card.append(parent_card)
        cpd = combine_cpd(node, card, parents, parents_card, bn_list, cpd_variant)
        pooled_bn.add_cpds(cpd)
    print(pooled_bn.check_model())
    return pooled_bn


