from pgmpy.models import BayesianModel


def check_independencies_preservation(bn_list, pooled_bn: BayesianModel):
    indep = pooled_bn.get_independencies()
    indep = indep.closure()
    indep = indep.independencies
    final_ind = set(indep)
    common_independencies = set(bn_list[0].get_independencies().closure().independencies)
    for bn in bn_list:
        ind = set(bn.get_independencies().closure().independencies)
        common_independencies.intersection_update(ind)
    print(common_independencies)
    print(final_ind)
    print(common_independencies-final_ind)
    if common_independencies.issubset(final_ind):
        return True
    return False

