from pgmpy.independencies import Independencies
from pgmpy.estimators import PC
import random


def find_common_independencies(bn_list):
    common_independencies = set(bn_list[0].get_independencies().closure().independencies)
    for bn in bn_list:
        ind = set(bn.get_independencies().closure().independencies)
        common_independencies.intersection_update(ind)
    common_independencies = list(common_independencies)
    print(common_independencies)
    return Independencies(*common_independencies)


def pdag_to_dag(pdag, bn_list):
    dag = pdag.directed_edges
    for (u, v) in pdag.undirected_edges:
        a, b = 0, 0
        for bn in bn_list:
            if bn.has_edge(u, v):
                a += 1
            elif bn.has_edge(v, u):
                b += 1
        if a > b:
            dag.add((u, v))
        elif b > a:
            dag.add((v, u))
        else:
            if random.choice([True, False]):
                dag.add((u, v))
            else:
                dag.add((v, u))
    print(dag)
    graph_dict = {}
    for node in list(pdag.nodes):
        graph_dict[node] = set()
    for (e, v) in dag:
        graph_dict[v].add(e)
    return list(dag), graph_dict


def create_merged_dag(bn_list):
    """

    :param bn_list:
    :return:
    """
    ind = find_common_independencies(bn_list)
    pdag = PC(independencies=ind).estimate(ci_test="independence_match", return_type="pdag")
    dag, dag_dict = pdag_to_dag(pdag, bn_list)
    return pdag, dag, dag_dict
