from pgmpy.estimators import MaximumLikelihoodEstimator, MmhcEstimator, HillClimbSearch, BDeuScore
from pgmpy.inference import VariableElimination

from stage1 import create_merged_dag
from stage2 import add_merged_cpds, nx
from properties import check_independencies_preservation
from bns_definition import bn1, bn2, bn3, bn4, random_BN
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianModel
import pylab as plt


def train_bn(data):
    mmhc = MmhcEstimator(data)
    skeleton = mmhc.mmpc()
    print("Part 1) Skeleton: ", skeleton.edges())
    # use hill climb search to orient the edges:
    hc = HillClimbSearch(data)
    dag = hc.estimate(tabu_length=10, white_list=skeleton.to_directed().edges())
    print("Part 2) Model:    ", dag.edges())
    model = BayesianModel(dag)
    model.fit(data)
    return model


def test(model: BayesianModel, data):
    infer = VariableElimination(model)


# Create Source BNs
SIZE = 5000
SPLIT = 0.5
source = random_BN(5, 0.5)

# Sample data
inference = BayesianModelSampling(source)
data = inference.forward_sample(size=SIZE)
i = int(SIZE*SPLIT)
j = int(SIZE*0.2)
data1 = data[1000:4000]
data2 = data[4000:5000]
test_data = data[:1000]

# Learn BN
model = train_bn(data)
model1 = train_bn(data1)
model2 = train_bn(data2)

# Pooling
bn_list = [model1, model2]
dag, dag_dict = create_merged_dag(bn_list)
pooled_bn = add_merged_cpds(dag, dag_dict, bn_list)
print("Independencies preserved:")
print(check_independencies_preservation(bn_list, pooled_bn))

# Evaluate edges
all = set(source.edges)
correct = all.intersection(set(model.edges))
correct1 = all.intersection(set(model1.edges))
correct2 = all.intersection(set(model2.edges))
hamming_dag_score = len(correct) / len(all)
hamming_dag_score1 = len(correct1) / len(all)
hamming_dag_score2 = len(correct2) / len(all)
print(hamming_dag_score, hamming_dag_score1, hamming_dag_score2)

# Evaluate accuracy
# bn_list.append(pooled_bn)
scores = []
for m in [source, model, model1, model2]:
    infer = VariableElimination(m)
    correct, incorrect = 0, 0
    for i, test_case in test_data.iterrows():
        evidence = dict(test_case)
        y_true = evidence.pop('A')
        y_hat = infer.map_query(['A'], evidence=evidence)['A']
        if y_true == y_hat:
            correct += 1
        else:
            incorrect += 1
    accuracy = correct/(correct+incorrect)
    scores.append(accuracy)
for score in scores:
    print(f'Accuracy {score}')


# Evaluate BN similarity


# Show results
fig, axes = plt.subplots(nrows=1, ncols=5)
ax = axes.flatten()
nx.draw(source, with_labels=True, ax=ax[0], pos=nx.spring_layout(source))
ax[0].set_axis_off()
nx.draw(model, with_labels=True, ax=ax[1], pos=nx.spring_layout(model))
ax[1].set_axis_off()
nx.draw(model1, with_labels=True, ax=ax[2], pos=nx.spring_layout(model1))
ax[2].set_axis_off()
nx.draw(model2, with_labels=True, ax=ax[3], pos=nx.spring_layout(model2))
ax[3].set_axis_off()
nx.draw(pooled_bn, with_labels=True, ax=ax[4], pos=nx.spring_layout(pooled_bn))
ax[4].set_axis_off()
plt.show()






