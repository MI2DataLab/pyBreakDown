import numpy as np

class Explanation:
    def __init__(self, results, baseline):
        conts = np.array(results[:,2],dtype=float)
        cumulative = np.transpose(np.expand_dims(np.cumsum(conts),axis=0))
        self._results = np.concatenate((results, cumulative), axis=1) #attach new column
        self.final_row = (["Final prediction", np.sum(conts)])
        self._varval = [str(row[0])+"="+str(row[1]) for row in self._results]
    
    def __str__ (self):
        return '\n'.join([str(("Feature","Contribution"))]+[str(pair) for pair in zip(self._varval, self._results[:,2])])