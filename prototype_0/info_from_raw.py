# Author: Caitlin
# Using Sida's investigate_mi tool on raw data from our simulations 
import numpy as np
import pickle

import sys

from investigate_mi import investigate, plot_venn

def load_data(fn):
    with open(fn,'rb') as f:
        history = pickle.load(f)
    return history

def get_states(series, state='all', bins=100):
    series_discrete = (series * bins).astype(np.int)
    if state=='all':
        return series_discrete
    if state == 'pos': #x,y
        return series_discrete[:,:2]
    if state == 'angle_velocity':
        return series_discrete[:,2:]
    if state == 'x':
        return series_discrete[:,0]
    if state == 'y':
        return series_discrete[:,1]
    if state== 'angle':
        return series_discrete[:,2]
    if state == 'velocity':
        return series_discrete[:,3]

policy = sys.argv[1]
# policy = 'Policy_'
files = ['{}/{}_1000steps_0seed.p'.format(policy, policy), '{}/{}_10000steps_0seed.p'.format(policy, policy)]
steps = [1000, 10000]
for i,f in enumerate(files):

    data = load_data(f)

    granularities = [10, 100, 1000]
    states = ['x','y','angle','velocity', 'pos', 'angle_velocity']
    # states = ['velocity']

    # no moving window - history length is the entire sim
    # comparing two vehicle timeseries only

    num_vehicles = int(data.shape[1]/4)

    for bins in granularities:

        for state in states:
            Hy_given_x = []
            Hx_given_y = []
            MI_xy = []
            for v1_id in range(num_vehicles):

                for v2_id in range(num_vehicles):

                    if v1_id != v2_id:
                        # time series including all states
                        v1_ts = data[:,v1_id*4:v1_id*4+4]

                        v2_ts = data[:,v2_id*4:v2_id*4+4]

                        v1_series = get_states(v1_ts, state=state, bins=bins)
                        v2_series = get_states(v2_ts, state=state, bins=bins)

                        # investigate(v1_series, v2_series, title='{}, {} steps, {} states, state={}, v1={}, v2={}'.format(policy, steps[i], bins, state, v1_id, v2_id))
                        info = investigate(v1_series, v2_series)
                        Hy_given_x.append(info["H(Y|X)"])
                        Hx_given_y.append(info["H(X|Y)"])
                        MI_xy.append(info["I(X;Y)"])
            
            Hy_given_x_avg = np.mean(Hy_given_x)
            Hx_given_y_avg = np.mean(Hx_given_y)
            MI_xy_avg = np.mean(MI_xy)

            title = "{}, {} steps, {} states, state={}, Average".format(policy, steps[i], bins, state)
            plot_venn(Hy_given_x_avg, Hx_given_y_avg, MI_xy_avg, title)


