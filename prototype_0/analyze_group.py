from dask import delayed
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
import glob
import pickle
import numpy as np
import scipy.stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

cluster = SLURMCluster(memory='2g',
                    cores=1,
                    queue='short',
                    walltime="03:00:00",
                    job_extra=['--job-name="simworker"', "--output=/users/d/m/dmatthe1/job_logs/dask-%x-%A.txt"])

cluster.scale(30)
client = Client(cluster)


def investigate(x, y):
    """x and y are observations of X and Y"""
    assert x.shape == y.shape, "Can't do mutual information on observations of different length"

    xy = np.c_[x, y]  # a faster way of doing xy = zip(x,y) and turn to array

    vals_x, counts_x = np.unique(x, return_counts=True, axis=0)
    vals_y, counts_y = np.unique(y, return_counts=True, axis=0)
    vals_xy, counts_xy = np.unique(xy, return_counts=True, axis=0)

    # H(X)
    Hx = scipy.stats.entropy(counts_x, base=2)
    # H(Y)
    Hy = scipy.stats.entropy(counts_y, base=2)
    # H(X,Y)
    Hxy = scipy.stats.entropy(counts_xy, base=2)
    # H(Y|X)
    Hy_given_x = Hxy - Hx
    # H(X|Y)
    Hx_given_y = Hxy - Hy
    # I(X;Y)
    MI_xy = Hy - Hy_given_x

    return (Hx, Hy, Hxy, Hy_given_x, Hx_given_y, MI_xy)


def process_data(fname, nbins):
    # read from disk
    data = pickle.load(open(fname, "rb"))
    
    # reshape to (time, vid, states)
    data = data.reshape((data.shape[0], -1, 4))
    
    # bin to nbins
    binned_data = (data * nbins).astype(np.int)

    velocity_binned_data = binned_data[:,:,2]
    n_vehicles = velocity_binned_data.shape[1]

    all_entropies = np.zeros(shape=((n_vehicles*(n_vehicles-1))//2, 7))
    row_id = 0
    seed_id = int(fname[fname.find("steps_")+6:-6])
    for v_id_a in range(n_vehicles):
        x_series = velocity_binned_data[:, v_id_a]
        for v_id_b in range(v_id_a+1, n_vehicles):
            y_series = velocity_binned_data[:, v_id_b]
            row_dat = investigate(x_series, y_series)
            all_entropies[row_id, 1:] = row_dat
            all_entropies[row_id, :1] = seed_id
            row_id += 1

    return all_entropies, np.mean(all_entropies, axis=0)

def main(client, fnames, nbins):
    results = []
    for fname in fnames:
        results.append(delayed(process_data)(fname, nbins))
    
    merged_data = []
    for fut in client.compute(results):
        res = fut.result()
        merged_data.append(res)
    
    return merged_data


policies = ["Policy", "Policy_Random", "Policy_Random_Network", "Policy_Random_Network2", "Policy_Follow_Leader", "Policy_Boids", "Policy_Simplified_Boids"]

dfs = []
for policy in policies:
    fnames = glob.glob("data/{}_10agents_10000steps*".format(policy))

    dat = main(client, fnames, 10)
    dat[0][1] # data from seed 1.

    average_entropies = np.array([d[1] for d in dat])
    average_entropies.shape

    df = pd.DataFrame(average_entropies, columns=["Seed", "Hx", "Hy", "Hxy", "Hy_given_x", "Hx_given_y", "MI_xy"])
    df.insert(0, "Policy", policy)
    dfs.append(df)
df = pd.concat(dfs)

ldf = pd.wide_to_long(df, stubnames=[""], i=["Policy", "Seed"], j="Metric", sep="", suffix='[HM]\w+')
ldf.reset_index(inplace=True)
ldf.rename(columns={"":"Value"}, inplace=True)

fig, ax = plt.subplots(figsize=(4*7,4))
sns.barplot(x="Policy", y="Value", hue="Metric",ci=95, data=ldf, ax=ax)
plt.savefig("Entropies.pdf")
plt.show()

tips = sns.load_dataset("tips")
tips