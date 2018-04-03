import numpy as np
import logging
#todo wrap in class
def fun (clf, observation, data, colnames):
    assert len(colnames) == data.shape[1]
    #go up
    if observation.ndim < 2:
        observation = np.expand_dims(observation, axis=0)
    new_data = np.repeat(observation,repeats=data.shape[0], axis=0)

    target_yhat = clf.predict(observation)
    baseline_yhat = np.mean(clf.predict(data))

    open_variables = list(range(0,data.shape[1]))
    important_variables = np.zeros(data.shape[1])
    important_yhats = {}

    for i in range(0, data.shape[1]):       
        yhats = {}
        yhats_diff = np.repeat(-float('inf'), data.shape[1])
        for variable in open_variables:
            current_data = data
            current_data[:,variable] = new_data[:,variable]
            yhats[variable] = clf.predict(current_data)
            yhats_diff[variable] = abs(baseline_yhat - np.mean(yhats[variable]))
        amax = np.argmax(yhats_diff)
        important_variables[i] = amax
        important_yhats[i] = yhats[amax]
        data[:,amax] = new_data[:,amax]
        open_variables.remove(amax) #todo - extremely ineffective - must rewrite

    iv = list(map(int,important_variables))
    var_names = np.expand_dims(np.array(colnames),axis=0)[0,iv]
    var_values = list(map(str, observation[:,iv]))
    means = list(map(np.mean, important_yhats))
    contributions = np.diff([baseline_yhat]+means)
    #todo return object
    return np.transpose(np.array([var_names, var_values, contributions]))

def test():
    from sklearn import datasets
    x = datasets.load_iris()
    data = x.data
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(data, x.target)
    return fun(clf, data[111,:], data, x.feature_names)