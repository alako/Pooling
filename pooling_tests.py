from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from stage1 import *
from stage2 import *
from pgmpy.inference import VariableElimination
import networkx as nx
import pylab as plt

# BN 1
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

# BN 2
# Defining the network structure
model2 = BayesianModel([('D', 'E'), ('D', 'A'), ('C', 'E')])
# Defining the CPDs:
# cpd_b = TabularCPD('A', 2, [[0.5, 0.5]])
cpd_b = TabularCPD('A', 2, [[0.4, 0.7],
                            [0.6, 0.3]],
                   evidence=['D'], evidence_card=[2])
cpd_c = TabularCPD('C', 2, [[0.3, 0.7]])
# cpd_d = TabularCPD('D', 2, [[0.4, 0.7],
#                             [0.6, 0.3]],
#                    evidence=['A'], evidence_card=[2])
cpd_d = TabularCPD('D', 2, [[0.4, 0.6]])
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

# Pooling
bn_list = [model1, model2]
dag, dag_dict = pool_graph(bn_list)
pooled_bn = add_cpds(dag, dag_dict, bn_list)

# Show results
nx.draw(pooled_bn, with_labels=True)
plt.show()
print(pooled_bn.get_cpds())





