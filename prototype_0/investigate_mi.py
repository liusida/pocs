# Author: Sida
# Show the complete Entropy Vann Diagram of two observations

import numpy as np
import scipy.stats
import pyinform as pyin
from matplotlib_venn import venn2
import matplotlib.pyplot as plt

def r(n):
    """venn2 can't show specified values, so round the values instead"""
    return round(n,3)

def investigate(x,y, title=""):
    """x and y are observations of X and Y"""
    assert x.shape==y.shape, "Can't do mutual information on observations of different length"

    xy = np.c_[x,y] # a faster way of doing xy = zip(x,y) and turn to array

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

    print( f"H(X): {Hx}" )
    print( f"H(Y): {Hy}")
    print( f"H(X,Y): {Hxy}" )
    print( f"H(Y|X): {Hy_given_x}")
    print( f"H(X|Y): {Hx_given_y}")
    print( f"I(X;Y): {MI_xy}")
    # In short:
    print( f"pyin.mutual_info: {pyin.mutual_info(x,y)}" )

    venn2(subsets = (r(Hy_given_x), r(Hx_given_y), r(MI_xy)), set_labels = ("H(X)", "H(Y)", "I(X;Y)"), normalize_to=1)
    plt.title(f"{title}")
    plt.savefig(f"investigate_{title}.png")
    plt.close()
    return {
        "H(X)": Hx,
        "H(Y)": Hy,
        "H(X,Y)": Hxy,
        "H(Y|X)": Hy_given_x,
        "H(X|Y)": Hx_given_y,
        "I(X;Y)": MI_xy,
    }

if __name__ == "__main__":
        
    x = [1,1,1,2,2,2,3]
    y = [1,2,1,2,1,2,3]
    x = np.array(x)
    y = np.array(y)
    # investigate(x,y)

    def two_iid_rv_of_100_possible_states():
        x = np.random.randint(low=0, high=100, size=[100000])
        y = np.random.randint(low=0, high=100, size=[100000])
        print(f"x: {x}")
        print(f"y: {y}")
        investigate(x,y,title="100k steps")

        x = np.random.randint(low=0, high=100, size=[1000])
        y = np.random.randint(low=0, high=100, size=[1000])
        print(f"x: {x}")
        print(f"y: {y}")
        investigate(x,y,title="1k steps")

        x = np.random.randint(low=0, high=100, size=[10])
        y = np.random.randint(low=0, high=100, size=[10])
        print(f"x: {x}")
        print(f"y: {y}")
        investigate(x,y,title="10 steps")

    def two_iid_rv_of_length_33():
        x = np.random.randint(low=0, high=2, size=[33])
        y = np.random.randint(low=0, high=2, size=[33])
        print(f"x: {x}")
        print(f"y: {y}")
        investigate(x,y,title="2 possible states")

        x = np.random.randint(low=0, high=4, size=[33])
        y = np.random.randint(low=0, high=4, size=[33])
        print(f"x: {x}")
        print(f"y: {y}")
        investigate(x,y,title="4 possible states")
    
        x = np.random.randint(low=0, high=8, size=[33])
        y = np.random.randint(low=0, high=8, size=[33])
        print(f"x: {x}")
        print(f"y: {y}")
        investigate(x,y,title="8 possible states")
        
    two_iid_rv_of_length_33()