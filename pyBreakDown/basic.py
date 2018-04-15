import numpy as np
import logging
from . import explainer as e

def test(dir, ind):
    from sklearn import datasets
    x = datasets.load_boston()
    data = x.data
    from sklearn import tree
    clf = tree.DecisionTreeRegressor()
    clf.fit(data, x.target)

    exp = e.Explainer(clf, x.feature_names)

    a = exp.explain(observation=data[ind,:], data=data, direction=dir, baseline=0)
    a.text()