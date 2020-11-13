import numpy as np
from  scipy import stats

class Metric:
    def __init__(self, world):
        self.world = world
        
    def get_metric(self):
        return self.world.vehicles[0].pos_x / self.world.width  # TODO: for demostration. Should be entropy or something.

class MicroEntropyMetric:
    def __init__(self, world):
        self.world = world
        self.history_len = 100
        self.world_history = np.zeros(shape=(self.history_len, len(self.world.vehicles), 4))
        
    
    def get_metric(self):
        # update the history information.
        if self.world_history.shape[1] != len(self.world.vehicles):
            self.world_history = np.zeros(shape=(self.history_len, len(self.world.vehicles), 4))
            self.history_idx = 0
        for v_id in range(self.world_history.shape[1]):
            self.world_history[self.history_idx, v_id] = [self.world.vehicles[v_id].pos_x, self.world.vehicles[v_id].pos_y,  self.world.vehicles[v_id].velocity,  self.world.vehicles[v_id].angle]
        self.history_idx += 1
        self.history_idx %= self.history_len
        
        total_micro_entropy = 0
        for v_id in range(self.world_history.shape[1]):
            # /10 bins the 1000x1000 world into 100 cells in x and y
            vals, counts = np.unique((self.world_history[:, v_id, 0]/10).astype(np.int), return_counts=True) # val counts of pos x.
            total_micro_entropy += stats.entropy(counts)
        return total_micro_entropy