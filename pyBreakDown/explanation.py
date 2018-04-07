import numpy as np

class Explanation:
    def __init__(self, results, baseline):
        conts = np.array(results[:,2],dtype=float)
        cumulative = np.transpose(np.expand_dims(np.cumsum(conts),axis=0))
        results = np.concatenate((results, cumulative), axis=1) #attach new column
        final_row = np.expand_dims(np.array(["","", np.sum(conts), np.sum(conts)]),axis=0)
        self._results = np.concatenate((results, final_row),axis=0)
        
    
    def __str__ (self):
        return str(self._results)