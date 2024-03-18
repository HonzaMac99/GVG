import numpy as np                # for matrix computation and linear algebra
import matplotlib.pyplot as plt   # for drawing and image I/O
import scipy
from scipy import linalg
from mpl_toolkits import mplot3d
import matplotlib.image as mpimg
# from PIL import Image
import scipy.io as sio            # for matlab file format output
import itertools                  # for generating all combinations


def e2p(u):
    n = u.shape[1]
    return np.vstack((u, np.ones((1, n))))


def p2e(p):
    if len(p.shape) < 2:
        p = np.vstack(p)  # correct 1d arrays to be vertical

    d = p.shape[0]

    assert d >= 3, "invalid dimension"
    assert p[d-1, :].all(), "division by zero"

    return p[:d-1, :] / p[d-1, :]


def p3p_distances(d12, d23, d31, c12, c23, c31):
    return
    # return N1, N2, N3


if __name__ == "__main__":
    K = sio.loadmat("K.mat")
    K = K['K']  # get the np array from dict
