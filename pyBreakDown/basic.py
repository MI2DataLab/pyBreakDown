import numpy as np
import logging
from . import explainer as e

def test(dir):
    from sklearn import datasets
    x = datasets.load_boston()
    data = x.data
    from sklearn import tree
    clf = tree.DecisionTreeRegressor()
    clf.fit(data, x.target)

    exp = e.Explainer(clf, x.feature_names)

    print(exp.explain(observation=data[111,:], data=data, direction=dir, baseline=0))