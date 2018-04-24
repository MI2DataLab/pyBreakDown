import numpy as np
from . import explainer as e

def test(dir, ind):
    from sklearn import datasets
    x = datasets.load_boston()
    data = x.data
    from sklearn import tree
    clf = tree.DecisionTreeRegressor()
    clf.fit(data, x.target)

    exp = e.Explainer(clf, data, x.feature_names)

    a = exp.explain(observation=data[ind,:], direction=dir, baseline=0)
    a.text()
