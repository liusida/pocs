# Author: Caitlin
# Using Sida's investigate_mi tool on raw data from our simulations 
import numpy as np
import pickle

from investigate_mi import investigate

def load_data(fn):
    with open(fn,'rb') as f:
        history = pickle.load(f)
    return history

def get_states(series, state='all', bins=100):
    series_discrete = (series * bins).astype(np.int)
    if state=='all':
        # ???
        vals, counts = np.unique(series_discrete, return_counts=True, axis=0) 
        return counts
    if state == 'pos': #x,y
        # ???
        vals, counts = np.unique(series_discrete[:,:2], return_counts=True, axis=0) 
        return counts
    if state == 'x':
        return series_discrete[:,0]
    if state == 'y':
        return series_discrete[:,1]
    if state== 'angle':
        return series_discrete[:,2]
    if state == 'velocity':
        return series_discrete[:,3]


# files = ['Policy_Random_100steps_0seed.p', 'Policy_Random_1000steps_0seed.p', 'Policy_Random_10000steps_0seed.p']
files = ['Policy_Follow_Leader_10000steps_0seed.p']
policy = 'Policy_Follow_Leader'
steps = [10000]
for i,f in enumerate(files):

    data = load_data(f)

    bins=10
    # states = ['x','y','angle','velocity']
    states = ['velocity']

    # no moving window - history length is the entire sim
    # comparing two vehicle timeseries only

    v1_id = 1
    v2_id = 2

    # time series including all states
    v1_ts = data[:,v1_id*4:v1_id*4+4]

    v2_ts = data[:,v2_id*4:v2_id*4+4]

    for state in states:
        v1_series = get_states(v1_ts, state=state, bins=bins)
        v2_series = get_states(v2_ts, state=state, bins=bins)

        investigate(v1_series, v2_series, title='{}, {} steps, {} states, state={}'.format(policy, steps[i], bins, state))




