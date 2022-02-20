from stage1 import pool_graph
from stage2 import add_cpds
from bns_definition import bn1, bn2
import networkx as nx
import pylab as plt

# Create BNs
model1 = bn1()
model2 = bn2()

# Pooling
bn_list = [model1, model2]
dag, dag_dict = pool_graph(bn_list)
pooled_bn = add_cpds(dag, dag_dict, bn_list)

# Show results
nx.draw(pooled_bn, with_labels=True)
plt.show()
print(pooled_bn.get_cpds())





