import numpy as np
from collections import deque
from blist import blist
from . import explanation

class Explainer:
    def __init__(self, clf, colnames):
        self.colnames = colnames
        self.clf = clf

    def _transform_observation (self, observation):
        if observation.ndim < 2:
            observation = np.expand_dims(observation, axis=0)
        return observation

    def _get_initial_dataset(self, observation, data):
        assert observation.ndim == 2 and observation.shape[0] == 1
        return np.repeat(observation,repeats=data.shape[0], axis=0)

    def explain (self, observation, data, direction, useIntercept = False, baseline=0):
        assert direction in ["up","down"]
        if direction=="up":
            results = self._explain_up(observation, baseline, np.copy(data))
        if direction=="down":
            results = self._explain_down(observation, baseline, np.copy(data))

        meanpred = np.mean(self.clf.predict(data))

        if useIntercept:
            baseline = meanpred
            bcont = 0
        else:
            bcont = meanpred - baseline
        
        intercept_row = np.expand_dims(np.array(["Intercept", 1, bcont]),axis=0)
        results = np.concatenate((intercept_row, results),axis=0)
        return explanation.Explanation(results, baseline)


    def _explain_up (self, observation, baseline, data):
        #go up
        assert len(self.colnames) == data.shape[1]
        #todo handle differences between observation and data feature spaces?
        observation = self._transform_observation(observation)
        assert observation.shape[1] == data.shape[1]
        new_data = self._get_initial_dataset(observation, data)

        baseline_yhat = np.mean(self.clf.predict(data))

        open_variables = blist(range(0,data.shape[1]))
        important_variables = deque()
        important_yhats = {}

        for i in range(0, data.shape[1]):       
            yhats = {}
            yhats_diff = np.repeat(-float('inf'), data.shape[1])
            
            for variable in open_variables:
                tmp_data = np.copy(data)
                tmp_data[:,variable] = new_data[:,variable]
                yhats[variable] = self.clf.predict(tmp_data)
                yhats_diff[variable] = abs(baseline_yhat - np.mean(yhats[variable]))

            amax = np.argmax(yhats_diff)
            important_variables.append(amax)
            important_yhats[i] = yhats[amax]
            data[:,amax] = new_data[:,amax]
            open_variables.remove(amax)

        var_names = np.array(self.colnames)[important_variables]
        var_values = observation[0,important_variables]
        means = deque([np.array(v).mean() for k, v in important_yhats.items()])
        means.appendleft(baseline_yhat)
        contributions = np.diff(means)
        return np.transpose(np.array([var_names, var_values, contributions]))
    
    def _explain_down (self, observation, baseline, data):
        #go down
        assert len(self.colnames) == data.shape[1]
        #todo handle differences between observation and data feature spaces?
        observation = self._transform_observation(observation)
        assert observation.shape[1] == data.shape[1]
        new_data = self._get_initial_dataset(observation, data)

        target_yhat = self.clf.predict(observation)

        open_variables = blist(range(0,data.shape[1]))
        important_variables = deque()
        important_yhats = {}

        for i in range(0, data.shape[1]):       
            yhats = {}
            yhats_diff = np.repeat(float('inf'), data.shape[1])
            
            for variable in open_variables:
                tmp_data = np.copy(new_data)
                tmp_data[:,variable] = data[:,variable]
                yhats[variable] = self.clf.predict(tmp_data)
                yhats_diff[variable] = abs(target_yhat - np.mean(yhats[variable]))

            amin = np.argmin(yhats_diff)
            important_variables.append(amin)
            important_yhats[i] = yhats[amin]
            new_data[:,amin] = data[:,amin]
            open_variables.remove(amin)

        important_variables.reverse()
        var_names = np.array(self.colnames)[important_variables]
        var_values = observation[0,important_variables]
        means = deque([np.array(v).mean() for k, v in important_yhats.items()])
        means.appendleft(target_yhat[0])
        means.reverse()
        contributions = np.diff(means)
        return np.transpose(np.array([var_names, var_values, contributions]))
