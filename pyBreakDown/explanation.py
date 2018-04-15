import numpy as np
from collections import deque

class Explanation:

    _INTERCEPT_NAME = "Intercept"
    _INTERCEPT_VALUE = 1

    def __init__(self, variable_names, variable_values, contributions):
        self._variable_names = deque(variable_names)
        self._variable_values = deque(variable_values)
        self._contributions = deque(contributions)
    
    def text (self, fwidth=15, contwidth=10, cumulwidth = 10):
        assert len(self._variable_names) == len(self._variable_values)
        assert len(self._variable_names) == len(self._contributions)
        assert len(self._variable_names) == len(self._cumulative)
        feats = [' = '.join([name, str(val)]) for name, val in zip (self._variable_names, self._variable_values)]
        lines = [
            ''.join(
            [feats[i].ljust(fwidth), self._contributions[i].ljust(contwidth), self._cumulative[i].ljust(cumulwidth)]) 
            for i in range(0,len(self._contributions))
            ]
        print (''.join(["Feature".ljust(fwidth), "Contribution".ljust(contwidth), "Cumulative".ljust(cumulwidth)]))
        print('\n'.join(lines))
        print(''.join(['Final prediction'.ljust(fwidth+contwidth), str(self._final_prediction).ljust(cumulwidth)]))
        print(' = '.join(["Baseline", str(self._baseline)]))

    def add_intercept (self, intercept_contribution):
        self._variable_names.appendleft(self._INTERCEPT_NAME)
        self._variable_values.appendleft(self._INTERCEPT_VALUE)
        self._contributions.appendleft(intercept_contribution)

    def make_final_prediction (self):
        self._cumulative = deque(np.cumsum(self._contributions))
        self._final_prediction = sum(self._contributions)

    def add_baseline (self, baseline):
        self._baseline = baseline