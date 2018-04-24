import numpy as np
from collections import deque

class Explanation:
    """
    Contains algorithm results, including contribiutions of each individual features.
    """
    _INTERCEPT_NAME = "Intercept"
    _INTERCEPT_VALUE = 1

    def __init__ (self):
        self._variable_names = deque()
        self._variable_values = deque()
        self._contributions = deque()

    def add_results(self, variable_names, variable_values, contributions):
        self._variable_names.append(variable_names)
        self._variable_values.append(variable_values)
        self._contributions.append(contributions)
    
    def text (self, fwidth=25, contwidth=20, cumulwidth = 20, digits=2):
        """
        Get user-friendly text from of explanation

        Parameters
        ----------
        fwidth : int
            Width of column with feature names, in digits.
        contwidth : int
            Width of column with contributions, in digits.
        cumulwidth : int
            Width of column with cumulative values, in digits.
        digits : int
            Number of decimal places for values.

        """

        assert len(self._variable_names) == len(self._variable_values)
        assert len(self._variable_names) == len(self._contributions)
        assert len(self._variable_names) == len(self._cumulative)
        feats = [
            ' = '.join([name, str(val)]) 
            for name, val 
                in zip (self._variable_names, self._variable_values)
                ]
        lines = [
            ''.join(
            [feats[i].ljust(fwidth), 
            str(round(self._contributions[i],digits)).ljust(contwidth), 
            str(round(self._cumulative[i], digits)).ljust(cumulwidth)]
            ) 
            for i in range(0,len(self._contributions))
                ]

        print (''.join(
            ["Feature".ljust(fwidth), 
            "Contribution".ljust(contwidth), 
            "Cumulative".ljust(cumulwidth)]))
        print('\n'.join(lines))
        print(''.join(
            ['Final prediction'.ljust(fwidth+contwidth), 
            str(round(self._final_prediction, digits)).ljust(cumulwidth)]))
        print(' = '.join(["Baseline", str(round(self._baseline, digits))]))

    def add_intercept (self, intercept_contribution):
        self._variable_names.appendleft(self._INTERCEPT_NAME)
        self._variable_values.appendleft(self._INTERCEPT_VALUE)
        self._contributions.appendleft(intercept_contribution)

    def make_final_prediction (self):
        self._cumulative = deque(np.cumsum(self._contributions))
        self._final_prediction = sum(self._contributions)

    def add_baseline (self, baseline):
        self._baseline = baseline