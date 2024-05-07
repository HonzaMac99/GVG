import numpy as np                # for matrix computation and linear algebra
import matplotlib.pyplot as plt   # for drawing and image I/O
from mpl_toolkits import mplot3d  # for 3d plots
import matplotlib.image as mpimg
import scipy
import scipy.linalg
import scipy.io as sio            # for matlab file format output
import itertools                  # for generating all combinations
from PIL import Image

from hw08 import u2F
from plot import *


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


def norm(p):
    return e2p(p2e(p))


def cross(a, b):
    return np.vstack(np.cross(a.flatten(), b.flatten()))


def get_from_dict(dict, ps):
    return [dict[p] for p in ps]


def compute_errs(u1, u2, F):
    l1 = F.T @ e2p(u2)
    l2 = F @ e2p(u1)

    # compute distances from lines for both imgs
    d1 = np.abs(np.sum(l1*e2p(u1), axis=0) / np.sqrt(l1[0]**2 + l1[1]**2))
    d2 = np.abs(np.sum(l2*e2p(u2), axis=0) / np.sqrt(l2[0]**2 + l2[1]**2))

    ep_errors = [d1, d2]
    return ep_errors


def get_best_Fe(u1, u2, ix):
    Fe_best = np.zeros((3, 3))
    e_max_best = np.inf
    err_matches = []
    points_sel = None
    iter = itertools.combinations(range(0, 12), 7)  # every 7 pts of 10 pts
    for inx in iter:
        ix_sel = ix[np.array(inx)]
        u1_sel = u1[:, ix_sel]
        u2_sel = u2[:, ix_sel]

        FF = u2F(u1_sel, u2_sel)
        for F in FF:
            if np.iscomplex(F).any() or np.linalg.matrix_rank(F) != 2:
                continue
            F = np.real(F)

            # change F --> Fe so that we work with corrected E
            E = K.T @ F @ K
            [U, S, Vt] = np.linalg.svd(E)
            D = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]])
            K_inv = np.linalg.inv(K)
            Ex = U @ D @ Vt
            Fe = K_inv.T @ Ex @ K_inv

            # compute distances from lines for both imgs
            [d1, d2] = compute_errs(u1, u2, Fe)

            ep_errors = d1 + d2
            e_max = np.max(ep_errors)

            if e_max < e_max_best:
                print("New best e:", e_max)
                e_max_best = e_max
                Fe_best = Fe
                points_sel = ix_sel
                err_matches = [d1, d2]

    return Fe_best, err_matches, points_sel


if __name__ == "__main__":
    img1 = mpimg.imread("data/daliborka_01.jpg")
    img2 = mpimg.imread("data/daliborka_23.jpg")
    data_hw8 = sio.loadmat("data/08_data.mat")

    [u1, u2, points_sel, ix, F] = get_from_dict(data_hw8, ('u1', 'u2', 'point_sel', 'ix', 'F'))
    K = np.array(sio.loadmat("data/K.mat")["K"])

    # === step 1 ===
    E1 = K.T @ F @ K
    [U, S, Vt] = np.linalg.svd(E1)
    D = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0]])
    K_inv = np.linalg.inv(K)
    Ex = U @ D @ Vt
    Fx = K_inv.T @ Ex @ K_inv

    errs = compute_errs(u1, u2, Fx)
    ix = ix.flatten()

    # === steps 2-3 ===
    plot_ep_lines(img1, img2, u1, u2, ix, Fx, "09_egx.pdf")
    plot_ep_errors(errs, "09_errorsx.pdf")

    # === step 4 ===
    # compute F with the E constraint
    Fe, errs, points_sel_e = get_best_Fe(u1, u2, ix)
    E = K.T @ Fe @ K

    # === steps 5-6 ===
    plot_ep_lines(img1, img2, u1, u2, ix, Fe, "09_eg.pdf")
    plot_ep_errors(errs, "09_errors.pdf")

    # === step 7 ===
    sio.savemat('09a_data.mat', {
        'F': F,
        'Ex': Ex,
        'Fx': Fx,
        'E': E,
        'Fe': Fe,
        'u1': u1,
        'u2': u2,
        'point_sel_e': points_sel_e,
    })
