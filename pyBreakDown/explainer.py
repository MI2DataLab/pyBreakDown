import numpy as np
from collections import deque
from blist import blist
from pyBreakDown import explanation as e
from tqdm import tqdm


class Explainer:
    """
    Explainer object.

    Parameters
    ----------
        clf : np.array
            Sklearn predicition model (regression or classification).
        data : np.array
            Baseline dataset for algorithm.
        colnames : np.array
            Dataset feature names.
    """
    def __init__(self, clf, data, colnames):
        assert len(colnames) == data.shape[1]  # otherwise it wouldn't make any sense
        self.clf = clf
        self.data = data
        self.colnames = colnames

    @staticmethod
    def _transform_observation (observation):
        if observation.ndim < 2:
            observation = np.expand_dims(observation, axis=0)
        return observation

    @staticmethod
    def _get_initial_dataset(observation, data):
        assert observation.ndim == 2 and observation.shape[0] == 1
        return np.repeat(observation,repeats=data.shape[0], axis=0)

    def explain (self, observation, direction, useIntercept = False, baseline=0):
        """
        Make explanation for given observation and dataset.

        Method works with any sklearn prediction model

        Parameters
        ----------
        observation : np.array
            Observation to explain.
        direction : str
            Could be "up" or "down". Decides the direction of algorithm.
        useIntercept : bool
            If set, baseline argument will be ignored and baseline will be set to intercept.
        baseline : float
            Baseline of explanation.

        Returns
        -------
        Explanation
            Object that contains influences and descriptions of each relevant attribute.

        """
        data = np.copy(self.data)
        assert direction in ["up", "down"]
        observation = self._transform_observation(observation)  # expand dims from 1D to 2D if necessary
        assert len(self.colnames) == observation.shape[1]

        if direction == "up":
            exp = self._explain_up(observation, data)
        else:
            exp = self._explain_down(observation, data)

        if hasattr(self.clf, 'predict_proba'):
            mean_prediction = np.mean(self.clf.predict_proba(data)[:, 0])
        else:
            mean_prediction = np.mean(self.clf.predict(data))

        if useIntercept:
            baseline = mean_prediction
            bcont = 0
        else:
            bcont = mean_prediction - baseline
        
        exp.add_intercept(bcont)
        exp.add_baseline(baseline)
        exp.make_final_prediction()
        return exp

    def _explain_up(self, observation, data):
        new_data = self._get_initial_dataset(observation, data)

        if hasattr(self.clf, 'predict_proba'):
            baseline_yhat = np.mean(self.clf.predict_proba(data)[:, 0])
            open_variables = blist(range(0, data.shape[1]))
            important_variables = deque()
            important_yhats = {}
            temp = np.copy(data)

            for i in tqdm(range(0, data.shape[1])):
                yhats = {}
                yhats_diff = np.repeat([-float('inf')], data.shape[1])

                for variable in open_variables:
                    data[:, variable] = new_data[:, variable]
                    yhats[variable] = self.clf.predict_proba(data)[:, 0]
                    yhats_diff[variable] = abs(baseline_yhat - np.mean(yhats[variable]))
                    data[:, variable]=temp[:, variable]

                amax = np.argmax(yhats_diff)
                important_variables.append(amax)
                important_yhats[i] = yhats[amax]
                data[:, amax] = new_data[:, amax]
                open_variables.remove(amax)
        else:
            baseline_yhat = np.mean(self.clf.predict(data))
            open_variables = blist(range(0, data.shape[1]))
            important_variables = deque()
            important_yhats = {}
            temp = np.copy(data)

            for i in tqdm(range(0, data.shape[1])):
                yhats = {}
                yhats_diff = np.repeat([-float('inf')], data.shape[1])

                for variable in open_variables:
                    data[:,variable] = data[:,variable]
                    yhats[variable] = self.clf.predict(data)[:, 0]
                    yhats_diff[variable] = abs(baseline_yhat - np.mean(yhats[variable]))
                    data[:, variable] = temp[:, variable]

                amax = np.argmax(yhats_diff)
                important_variables.append(amax)
                important_yhats[i] = yhats[amax]
                data[:,amax] = new_data[:,amax]
                open_variables.remove(amax)

        var_names = np.array(self.colnames)[important_variables]
        var_values = observation[0, important_variables]
        means = self._get_means_from_yhats(important_yhats)
        means.appendleft(baseline_yhat)
        contributions = np.diff(means)
        return e.Explanation(var_names, var_values, contributions, e.ExplainerDirection.Up)

    def _explain_down (self, observation, data):
        new_data = self._get_initial_dataset(observation, data)

        if hasattr(self.clf, 'predict_proba'):
            target_yhat = self.clf.predict_proba(observation)[:, 0]

            open_variables = blist(range(0, data.shape[1]))
            important_variables = deque()
            important_yhats = {}
            temp = np.copy(new_data)

            for i in range(0, data.shape[1]):
                yhats = {}
                yhats_diff = np.repeat([float('inf')], data.shape[1])

                for variable in open_variables:
                    new_data[:,variable] = data[:,variable]
                    yhats[variable] = self.clf.predict_proba(new_data)[:, 0]
                    yhats_diff[variable] = abs(target_yhat - np.mean(yhats[variable]))
                    new_data[:, variable] = temp[:, variable]

                amin = np.argmin(yhats_diff)
                important_variables.append(amin)
                important_yhats[i] = yhats[amin]
                new_data[:,amin] = data[:,amin]
                open_variables.remove(amin)
        else:
            target_yhat = self.clf.predict(observation)[:, 0]

            open_variables = blist(range(0, data.shape[1]))
            important_variables = deque()
            important_yhats = {}
            temp = np.copy(new_data)

            for i in range(0, data.shape[1]):
                yhats = {}
                yhats_diff = np.repeat([float('inf')], data.shape[1])

                for variable in open_variables:
                    new_data[:, variable] = data[:, variable]
                    yhats[variable] = self.clf.predict(new_data)[:, 0]
                    yhats_diff[variable] = abs(target_yhat - np.mean(yhats[variable]))
                    new_data[:, variable] = temp[:, variable]

                amin = np.argmin(yhats_diff)
                important_variables.append(amin)
                important_yhats[i] = yhats[amin]
                new_data[:, amin] = data[:, amin]
                open_variables.remove(amin)

        important_variables.reverse()
        var_names = np.array(self.colnames)[important_variables]
        var_values = observation[0,important_variables]
        means = self._get_means_from_yhats(important_yhats)
        means.appendleft(target_yhat[0])
        means.reverse()
        contributions = np.diff(means)
        
        return e.Explanation(var_names, var_values, contributions, e.ExplainerDirection.Down)

    @staticmethod
    def _get_means_from_yhats (important_yhats):
        return deque([np.array(v).mean() for k, v in important_yhats.items()])
