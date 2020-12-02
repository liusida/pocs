# Author: Atoosa
# Description:
# This is an implementation of mutual information
# References:
#   Book: Elements of Information Theory
#   https://www.roelpeters.be/calculating-mutual-information-in-python/
#   https://elife-asu.github.io/PyInform/timeseries.html#module-pyinform.mutualinfo

# Used a lot of David's code from metric.py

import numpy as np
import pyinform as pyin
from  scipy import stats
from metric import calc_entropy

def relativeEntropy(p, q):
    return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)))

class MutualInfoMetric:
    def __init__(self, world, **kwargs):
        self.world = world
        self.history_len = kwargs["history_len"] if "history_len" in kwargs else 100
        self.world_history = np.zeros(shape=(self.history_len, len(self.world.vehicles), 4))
        self.history_idx = 0
        self.grid_size = 10
        
    def update_history(self):
        # update the history information.
        if self.world_history.shape[1] != len(self.world.vehicles):
            self.world_history = np.zeros(shape=(self.history_len, len(self.world.vehicles), 4))
            self.history_idx = 0
        world_width = self.world.width
        world_height = self.world.height
        world_size = min(world_width, world_height)
        for v_id in range(self.world_history.shape[1]):
            # save normalized metrics. between 0 and 1.
            self.world_history[self.history_idx, v_id] = [self.world.vehicles[v_id].pos_x /world_width,
                                                            self.world.vehicles[v_id].pos_y/ world_height,
                                                            self.world.vehicles[v_id].velocity / world_size,
                                                            self.world.vehicles[v_id].angle / (np.pi * 2)]
        self.history_idx += 1
        self.history_idx %= self.history_len
        
    def get_metric(self):
        self.update_history()
        
        total_mi = 0
        total_mi2 = 0
        for v_id_1 in range(self.world_history.shape[1]):
            for v_id_2 in range(self.world_history.shape[1]):
                if (v_id_1 == v_id_2):
                    continue
                # save and bin to ints.
                row_history_1 = (self.world_history[:, v_id_1, 3] * self.grid_size).astype(np.int)
                #print(row_history_1)
                #print(len(row_history_1))
                # vals_1, counts_1 = np.unique(row_history_1[:,3], return_counts=True, axis=0) # val counts of state history of one particle
                #print(counts_1)
                # save and bin to ints.
                row_history_2 = (self.world_history[:, v_id_2, 3] * self.grid_size).astype(np.int)
                # vals_2, counts_2 = np.unique(row_history_2[:,3], return_counts=True, axis=0) # val counts of state history of one particle
                
                curr_mi = pyin.mutual_info(row_history_1.flatten(), row_history_2.flatten())
                # curr_mi /= (np.log2( 1/self.history_len)* -1) # normalize to 0 to 1
                curr_mi2  = curr_mi + 1e-4
                curr_mi2 /= (pyin.shannon.entropy(pyin.Dist(row_history_1.flatten())) + 
                        pyin.shannon.entropy(pyin.Dist(row_history_2.flatten())) +
                        1e-4)
                total_mi += curr_mi
                total_mi2 += curr_mi2
                
        return {"Mutual Information": total_mi / (self.world_history.shape[1]**2)} #,
#                "MI normalized": total_mi2 / (self.world_history.shape[1]**2)} # normalize to 0 to 1