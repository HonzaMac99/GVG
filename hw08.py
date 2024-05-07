import numpy as np                # for matrix computation and linear algebra
import matplotlib.pyplot as plt   # for drawing and image I/O
from mpl_toolkits import mplot3d  # for 3d plots
import matplotlib.image as mpimg
import scipy
import scipy.linalg
import scipy.io as sio            # for matlab file format output
import itertools                  # for generating all combinations
from PIL import Image

from plot import *


# u2f
def u2F_polynom( G1, G2 ):
    a3 = np.linalg.det( G2 )

    a2 = (G2[1, 0] * G2[2, 1] * G1[0, 2]
          - G2[1, 0] * G2[0, 1] * G1[2, 2]
          + G2[0, 0] * G2[1, 1] * G1[2, 2]
          + G2[2, 0] * G1[0, 1] * G2[1, 2]
          + G2[2, 0] * G2[0, 1] * G1[1, 2]
          - G2[0, 0] * G1[2, 1] * G2[1, 2]
          - G2[2, 0] * G1[1, 1] * G2[0, 2]
          - G2[2, 0] * G2[1, 1] * G1[0, 2]
          - G2[0, 0] * G2[2, 1] * G1[1, 2]
          + G1[1, 0] * G2[2, 1] * G2[0, 2]
          + G2[1, 0] * G1[2, 1] * G2[0, 2]
          + G1[2, 0] * G2[0, 1] * G2[1, 2]
          - G1[1, 0] * G2[0, 1] * G2[2, 2]
          - G1[0, 0] * G2[2, 1] * G2[1, 2]
          - G2[1, 0] * G1[0, 1] * G2[2, 2]
          + G2[0, 0] * G1[1, 1] * G2[2, 2]
          + G1[0, 0] * G2[1, 1] * G2[2, 2]
          - G1[2, 0] * G2[1, 1] * G2[0, 2])

    a1 = (G1[0, 0] * G1[1, 1] * G2[2, 2]
          + G1[0, 0] * G2[1, 1] * G1[2, 2]
          + G2[2, 0] * G1[0, 1] * G1[1, 2]
          - G1[1, 0] * G1[0, 1] * G2[2, 2]
          - G2[0, 0] * G1[2, 1] * G1[1, 2]
          - G2[1, 0] * G1[0, 1] * G1[2, 2]
          - G2[2, 0] * G1[1, 1] * G1[0, 2]
          + G2[0, 0] * G1[1, 1] * G1[2, 2]
          + G1[1, 0] * G1[2, 1] * G2[0, 2]
          + G1[1, 0] * G2[2, 1] * G1[0, 2]
          + G1[2, 0] * G2[0, 1] * G1[1, 2]
          - G1[1, 0] * G2[0, 1] * G1[2, 2]
          - G1[2, 0] * G2[1, 1] * G1[0, 2]
          + G2[1, 0] * G1[2, 1] * G1[0, 2]
          - G1[0, 0] * G2[2, 1] * G1[1, 2]
          - G1[2, 0] * G1[1, 1] * G2[0, 2]
          + G1[2, 0] * G1[0, 1] * G2[1, 2]
          - G1[0, 0] * G1[2, 1] * G2[1, 2])

    a0 = np.linalg.det( G1 )

    return [a0, a1, a2, a3]


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


def u2F(u1, u2):
    '''Compute F using the seven-point algorithm from 7 euclidean correspondences u1, u2'''

    u_1, v_1 = u1[0], u1[1]
    u_2, v_2 = u2[0], u2[1]

    # A = [u2*u1, u2*v1, u2*w1, v2*u1, v2*v1, v2*w1, w2*u1, w2*v1, w2*w1] --> ws are 1 here
    A = np.c_[u_2*u_1, u_2*v_1, u_2, v_2*u_1, v_2*v_1, v_2, u_1, v_1, np.ones(7)]

    [U, S, Vt] = np.linalg.svd(A)
    G1 = Vt[-2, :].reshape(3, 3)
    G2 = Vt[-1, :].reshape(3, 3)
    # G1, G2 = scipy.linalg.null_space(A).T  # returns last two rows of Vt

    pol_coefs = u2F_polynom(G1, G2)
    pol_coefs.reverse()
    alphas = np.roots(pol_coefs)
    FF = []
    for alpha in alphas:
        G = G1 + alpha*G2
        if not np.iscomplex(G).any() and np.linalg.matrix_rank(G) == 2 and not np.allclose(G, G2):
            FF.append(G)
    return FF


def get_best_F(u1, u2, ix):
    F_best = np.zeros((3, 3))
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
            l1 = F.T @ e2p(u2)
            l2 = F @ e2p(u1)

            # compute distances from lines for both imgs
            d1 = np.abs(np.sum(l1*e2p(u1), axis=0) / np.sqrt(l1[0]**2 + l1[1]**2))
            d2 = np.abs(np.sum(l2*e2p(u2), axis=0) / np.sqrt(l2[0]**2 + l2[1]**2))

            ep_errors = d1 + d2
            e_max = np.max(ep_errors)

            if e_max < e_max_best:
                print("New best e:", e_max)
                e_max_best = e_max
                F_best = F
                points_sel = ix_sel
                err_matches = [d1, d2]

    return F_best, err_matches, points_sel


if __name__ == "__main__":
    # mistake 1: thought that crp_idx are the actual correspondence indexes, so
    #            I was picking totaly different points! [189]
    #            --> next time check by plotting the points first, if there is uncertainty
    # mistake 2: didn't subtract 1 from ix (in matlab format) [188]
    # mistake 3: wrong coef. order in np.roots(polynom) [132]

    img1 = mpimg.imread("data/daliborka_01.jpg")
    img2 = mpimg.imread("data/daliborka_23.jpg")
    data = sio.loadmat("data/daliborka_01_23-uu.mat")
    K = np.array(sio.loadmat("data/K.mat"))

    ix = data['ix'].flatten() - 1
    # crp_idx = data['edges'][:, ix]
    u1 = data['u01']
    u2 = data['u23']
    n_points = u2.shape[1]

    F, errs, points_sel = get_best_F(u1, u2, ix)

    plot_ep_errors(errs, "08_errors.pdf")
    plot_ep_lines(img1, img2, u1, u2, ix, F, "08_eg.pdf")

    sio.savemat('08_data.mat', {
        'u1': u1,
        'u2': u2,
        'ix': ix,
        'point_sel': points_sel,
        'F': F
    })
