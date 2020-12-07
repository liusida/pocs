# Author: Caitlin
# Using Sida's investigate_mi tool on raw data from our simulations 
import numpy as np
import pickle
import matplotlib.pyplot as plt

import sys

from investigate_mi import investigate, plot_venn


from matplotlib_venn import venn3, venn3_circles
# plt.figure(figsize=(4,4))
# v = venn3(subsets=(1, 1, 1, 1, 1, 1, 1), set_labels = ('A', 'B', 'C'))
# # v.get_patch_by_id('100').set_alpha(1.0)
# # v.get_patch_by_id('100').set_color('white')
# # v.get_label_by_id('100').set_text('Unknown')
# # v.get_label_by_id('A').set_text('Set "A"')
# # c = venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), linestyle='dashed')
# # c[0].set_lw(1.0)
# # c[0].set_ls('dotted')
# plt.title("Sample Venn diagram")
# plt.annotate('Unknown set', xy=v.get_label_by_id('100').get_position() - np.array([0, 0.05]), xytext=(-70,-70),
#              ha='center', textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
#              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',color='gray'))
# plt.show()

# exit()

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

policies = ['Policy', 'Policy_Random', 'Policy_Follow_Leader', 'Policy_Boids', 'Policy_Simplified_Boids', 'Policy_Random_Network', 'Policy_Random_Network2']
files = []
for p in policies:
    files.append('{}/{}_10000steps_0seed.p'.format(p, p))
steps = [10000]

# fig, ax = plt.subplots(len(policies),3, figsize=(10,15))
fig, ax = plt.subplots(6,len(policies), figsize=(15,13))

for i,f in enumerate(files):

    data = load_data(f)

    states = ['x','y','angle','velocity', 'pos', 'angle_velocity']
    # states = ['velocity']

    # no moving window - history length is the entire sim
    # comparing two vehicle timeseries only

    num_vehicles = int(data.shape[1]/4)
    granularities = [10]
    # granularities = [10, 100, 1000]

    for j,bins in enumerate(granularities):
        # print(policies[i], ":", bins, "bins")

        for k,state in enumerate(states):
            print(policies[i], ":", state)
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
                        info = investigate(v1_series, v2_series, plot=False)
                        Hy_given_x.append(info["H(Y|X)"])
                        Hx_given_y.append(info["H(X|Y)"])
                        MI_xy.append(info["I(X;Y)"])
            
            Hy_given_x_avg = np.mean(Hy_given_x)
            Hx_given_y_avg = np.mean(Hx_given_y)
            MI_xy_avg = np.mean(MI_xy)

            # title = "{}, {} steps, {} states, state={}, Average".format(policy, steps[i], bins, state)
            plot_venn(Hy_given_x_avg, Hx_given_y_avg, MI_xy_avg, ax=ax[k,i], with_numbers=False)


# cols = ["{} bins".format(g) for g in granularities]

cols = ['Policy', 'Random', 'Follow Leader', 'Boids', 'Simplified Boids', 'Random Network', 'Random Network 2']
rows = ['x','y','angle','velocity', '(x, y)', '(angle, velocity)']

for a, col in zip(ax[0], cols):
    a.set_title(col)

for a, row in zip(ax[:,0], rows):
    a.annotate(row, xy=(-.25,.5),                    
                xycoords='axes fraction',
                size='large', ha='right', va='center')

fig.tight_layout()
# plt.show()
plt.savefig('policy_state_VDs_10ksteps_10bins_without_numbers.png', dpi=300)
    
