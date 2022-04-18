from stage1 import create_merged_dag
from stage2 import add_merged_cpds
from properties import check_independencies_preservation
from bns_definition import bn1, bn2, bn3, bn4, random_BN
import networkx as nx
import pylab as plt

# Create BNs
model1 = random_BN(5, 0.5)
# model2 = bn3()
model2 = random_BN(5, 0.3)

# Pooling
bn_list = [model1, model2]
dag, dag_dict = create_merged_dag(bn_list)
pooled_bn = add_merged_cpds(dag, dag_dict, bn_list)
print("Independencies preserved:")
print(check_independencies_preservation(bn_list, pooled_bn))

# Show results
fig, axes = plt.subplots(nrows=1, ncols=3)
ax = axes.flatten()
nx.draw(model1, with_labels=True, ax=ax[0])
ax[0].set_axis_off()
nx.draw(model2, with_labels=True, ax=ax[1])
ax[1].set_axis_off()
nx.draw(pooled_bn, with_labels=True, ax=ax[2])
ax[2].set_axis_off()
plt.show()
print(pooled_bn.get_cpds())





