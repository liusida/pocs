# Author: Sida
# Show the complete Entropy Vann Diagram of two observations

from enum import unique
import numpy as np
import scipy.stats
import pyinform as pyin
from matplotlib_venn import venn2
import matplotlib.pyplot as plt


def r(n):
    """venn2 can't show specified values, so round the values instead"""
    return round(n, 3)


def investigate(x, y, title=""):
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

    print(f"H(X): {Hx}")
    print(f"H(Y): {Hy}")
    print(f"H(X,Y): {Hxy}")
    print(f"H(Y|X): {Hy_given_x}")
    print(f"H(X|Y): {Hx_given_y}")
    print(f"I(X;Y): {MI_xy}")
    # In short:
    print(f"pyin.mutual_info: {pyin.mutual_info(x,y)}")

    venn2(subsets=(r(Hy_given_x), r(Hx_given_y), r(MI_xy)), set_labels=("H(X)", "H(Y)", "I(X;Y)"), normalize_to=1)
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

