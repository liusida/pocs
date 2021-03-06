import numpy as np
from  scipy import stats
import pyinform as pyin
from sklearn.metrics import mutual_info_score

def calc_entropy(pk):
    if pk is None:
        return 0
    pk2 = np.array(pk)/np.sum(pk)
    return -np.sum(pk2 * np.log2(pk2))
class Metric:
    def __init__(self, world):
        self.world = world
        
    def get_metric(self):
        return {"pos": self.world.vehicles[0].pos_x}  # TODO: for demostration. Should be entropy or something.

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
        return {"Micro Entropy": total_micro_entropy / self.world_history.shape[1]} # normalize to 0 to 1

class MacroEntropyMetric(EntropyMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_size= kwargs["grid_size"] if "grid_size" in kwargs else 10

    def get_metric(self):
        self.update_history()

        if (self.world_history.shape[1] != 0):
            # world_pos_history = (self.world_history[:, :, :2] * self.grid_size).astype(np.int)
            world_pos_history = stats.rankdata(self.world_history[:, :, :2], axis=1) # compute the position rank tuple for each vehicle at each timestep.
            vals, counts = np.unique(world_pos_history, return_counts=True, axis=0) # val counts of pos x.
            entropy = calc_entropy(counts)
            return  {"Macro Entropy":entropy / (np.log2( 1/self.history_len)* -1)} # normalize to 0 to 1.
        else:
            return {"Macro Entropy":0}


class MacroMicroEntropyMetric(EntropyMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_size= kwargs["grid_size"] if "grid_size" in kwargs else 50

    def get_metric(self):
        self.update_history()
        
        macro_entropy = 0
        micro_entropy = 0
        if (self.world_history.shape[1] != 0):
            # world_pos_history = (self.world_history[:, :, :2] * self.grid_size).astype(np.int)
            world_pos_history = stats.rankdata(self.world_history[:, :, :2], axis=1) # compute the position rank tuple for each vehicle at each timestep.
            # print(world_pos_history[:10, :, 0])
            # print(world_pos_history[:10, :, 1])
            vals, counts = np.unique(world_pos_history, return_counts=True, axis=0) # val counts of pos x.
            # print(counts)
            macro_entropy = calc_entropy(counts)
            macro_entropy /= (np.log2( 1/self.history_len)* -1) # normalize to 0 to 1.


        total_micro_entropy = 0
        for v_id in range(self.world_history.shape[1]):
            # save and bin to ints.
            row_history = (self.world_history[:, v_id, 2:] * self.grid_size).astype(np.int)
            vals, counts = np.unique(row_history, return_counts=True, axis=0) # val counts of state history of one particle
            curr_entropy = calc_entropy(counts)
            curr_entropy /= (np.log2( 1/self.history_len)* -1) # normalize to 0 to 1
            total_micro_entropy += curr_entropy
        micro_entropy =  total_micro_entropy / self.world_history.shape[1] # normalize to 0 to 1

        # print("micro: %.2f, macro: %.2f"%(micro_entropy, macro_entropy))
        return {"Micro": micro_entropy,
                "Macro": macro_entropy,
                "Micro - Macro": micro_entropy - macro_entropy}

def get_world_wrapped(world, a, b):
    # print(a, b)
    if a < 0:
        a = world.shape[0] + a 
    if b < 0:
        b = world.shape[0] + b
    # print(a, b)
    if  b >= a:
        return world[a:b]
    else:
        a_arr = world[a:]
        b_arr = world[:b]
        # print("a_arr", a_arr.shape)
        # print("b_arr", b_arr.shape)
        return np.vstack([a_arr, b_arr])

class PredictiveInformationMetric(EntropyMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_history_len = kwargs["local_history_len"] if "local_history_len" in kwargs else int(1/3 * self.history_len)
        self.gap_history_len = kwargs["gap_history_len"] if "gap_history_len" in kwargs else int(1/3 * self.history_len)
        self.grid_size= kwargs["grid_size"] if "grid_size" in kwargs else 1000

    def get_metric(self):
        self.update_history()
        if (self.world_history.shape[1] != 0):
            
            total_pi_ang =  0
            total_pi_x =  0

            for v_id in range(self.world_history.shape[1]):
                prev_sub_world = get_world_wrapped(self.world_history, self.history_idx - self.local_history_len - self.gap_history_len, self.history_idx  - self.gap_history_len)
                curr_sub_world = get_world_wrapped(self.world_history, self.history_idx - self.local_history_len, self.history_idx)

                prev_history_angle = (prev_sub_world[:, v_id, 3] * self.grid_size).astype(np.int).flatten()
                curr_history_angle = (curr_sub_world[:, v_id, 3] * self.grid_size).astype(np.int).flatten()

                prev_history_x = (prev_sub_world[:, v_id, 0] * self.grid_size).astype(np.int).flatten()
                curr_history_x = (curr_sub_world[:, v_id, 0] * self.grid_size).astype(np.int).flatten()
            
                # print(prev_sub_world.shape, self.history_idx - self.local_history_len - self.gap_history_len, self.history_idx  - self.gap_history_len)
                # print(curr_sub_world.shape, self.history_idx - self.local_history_len, self.history_idx)

                # print(prev_vals, prev_counts)
                # print(curr_vals, curr_counts)
                entropy_ang = pyin.mutual_info(prev_history_angle, curr_history_angle)
                entropy_x = pyin.mutual_info(prev_history_x, curr_history_x)
                total_pi_ang += entropy_ang/ (np.log2( 1/self.history_len)* -1)
                total_pi_x += entropy_x/ (np.log2( 1/self.history_len)* -1)
            return  {"Predictive Micro Entropy Angle": total_pi_ang /self.world_history.shape[1],
                     "Predictive Micro Entropy X": total_pi_x /self.world_history.shape[1]} # normalize to 0 to 1.
        else:
            return {"Predictive Micro Entropy":0}