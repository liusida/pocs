import numpy as np
from  scipy import stats

def calc_entropy(pk):
    if pk is None:
        return 0
    pk2 = np.array(pk)/np.sum(pk)
    return -np.sum(pk2 * np.log2(pk2))
class Metric:
    def __init__(self, world):
        self.world = world
        
    def get_metric(self):
        return self.world.vehicles[0].pos_x  # TODO: for demostration. Should be entropy or something.

class EntropyMetric:
    def __init__(self, world, **kwargs):
        print(kwargs)
        self.world = world
        self.history_len = kwargs["history_len"] if "history_len" in kwargs else 100
        self.world_history = np.zeros(shape=(self.history_len, len(self.world.vehicles), 4))
        self.history_idx = 0
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
        raise NotImplementedError

class MicroEntropyMetric(EntropyMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_size= kwargs["grid_size"] if "grid_size" in kwargs else 100

    def get_metric(self):
        self.update_history()
        
        total_micro_entropy = 0
        for v_id in range(self.world_history.shape[1]):
            # save and bin to ints.
            row_history = (self.world_history[:, v_id] * self.grid_size).astype(np.int)
            vals, counts = np.unique(row_history[:,3], return_counts=True, axis=0) # val counts of state history of one particle
            curr_entropy = calc_entropy(counts)
            curr_entropy /= (np.log2( 1/self.history_len)* -1) # normalize to 0 to 1
            total_micro_entropy += curr_entropy
        return total_micro_entropy / self.world_history.shape[1] # normalize to 0 to 1

class MacroEntropyMetric(EntropyMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_size= kwargs["grid_size"] if "grid_size" in kwargs else 10

    def get_metric(self):
        self.update_history()

        if (self.world_history.shape[1] != 0):
            world_pos_history = (self.world_history[:, :, :2] * self.grid_size).astype(np.int)
            vals, counts = np.unique(world_pos_history, return_counts=True, axis=0) # val counts of pos x.
            entropy = calc_entropy(counts)
            return entropy / (np.log2( 1/self.history_len)* -1) # normalize to 0 to 1.
        else:
            return 0


class MacroMicroEntropyMetric(EntropyMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_size= kwargs["grid_size"] if "grid_size" in kwargs else 50

    def get_metric(self):
        self.update_history()
        
        macro_entropy = 0
        micro_entropy = 0
        if (self.world_history.shape[1] != 0):
            world_pos_history = (self.world_history[:, :, :2] * self.grid_size).astype(np.int)
            vals, counts = np.unique(world_pos_history, return_counts=True, axis=0) # val counts of pos x.
            macro_entropy = calc_entropy(counts)
            macro_entropy /= (np.log2( 1/self.history_len)* -1) # normalize to 0 to 1.


        total_micro_entropy = 0
        for v_id in range(self.world_history.shape[1]):
            # save and bin to ints.
            row_history = (self.world_history[:, v_id] * self.grid_size).astype(np.int)
            vals, counts = np.unique(row_history, return_counts=True, axis=0) # val counts of state history of one particle
            curr_entropy = calc_entropy(counts)
            curr_entropy /= (np.log2( 1/self.history_len)* -1) # normalize to 0 to 1
            total_micro_entropy += curr_entropy
        micro_entropy =  total_micro_entropy / self.world_history.shape[1] # normalize to 0 to 1

        print("micro: %.2f, macro: %.2f"%(micro_entropy, macro_entropy))
        return micro_entropy - macro_entropy
