# Author: Sida
# Show the complete Entropy Vann Diagram of two observations

import numpy as np
import scipy.stats
import pyinform as pyin
from matplotlib_venn import venn2
import matplotlib.pyplot as plt


def investigate(x,y):
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

    print( f"H(X): {Hx}, H(Y): {Hy}" )
    print( f"H(X,Y): {Hxy}" )
    print( f"H(Y|X): {Hy_given_x}")
    print( f"I(X;Y): {MI_xy}")
    # In short:
    print( f"pyin.mutual_info: {pyin.mutual_info(x,y)}" )

    venn2(subsets = (Hy_given_x, Hx_given_y, MI_xy), set_labels = ("H(X)", "H(Y)", "I(X;Y)"))
    plt.show()

x = [1,1,1,2,2,2]
y = [1,1,1,2,2,1]
x = np.array(x)
y = np.array(y)
investigate(x,y)