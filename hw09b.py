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


def E2RC(u1, u2, E):
    # first approach:
    M = E.T @ E                             # G.T@G / tau**2
    C_norm_2 = np.trace(M)                  # C_norm**2 = 2*(x**2 + y**2 + z**2)
    C_x = np.sqrt(-M[0, 0] + C_norm_2/2)    # x = sqrt(-(y**2 + z**2) + C_norm**2/2)
    C_y = -M[1, 0]/C_x                      # -(-xy)/x
    C_z = -M[2, 0]/C_x                      # -(-xz)/x
    C_norm = np.sqrt(C_norm_2)

    v1 = np.array([C_x, C_y, C_z]) / C_norm
    v1 = v1 / np.linalg.norm(v1)
    print("v from first approach: ", v1)

    # second approach
    E_stripe = E / C_norm                   # E_str = R @ [t]x
    [U, S, Vt] = np.linalg.svd(E_stripe)    # (R @ [t]x) * t = 0
    v2 = Vt[-1, :]
    v2 = v2 / np.linalg.norm(v2)
    print("v from second approach: ", v2)

    v = v2
    V = np.array([[    0,  v[2], -v[1]],
                  [-v[2],     0,  v[0]],
                  [ v[1], -v[0],     0]])

    g1, g2, g3 = E_stripe.T
    v1, v2, v3 = V.T
    Rs = []
    for s in [1, -1]:
        A = np.c_[s*g1, s*g2, s*g3, np.cross(g1, g2), np.cross(g2, g3), np.cross(g3, g1)]
        B = np.c_[  v1,   v2,   v3, np.cross(v1, v2), np.cross(v2, v3), np.cross(v3, v1)]

        # A = R@B --> B.T@R.T = A.T
        R = (np.linalg.lstsq(B.T, A.T, rcond=None)[0]).T
        # R = np.linalg.solve(B.T, A.T).T  # only for square matrices
        Rs.append(R)

    R_best = np.zeros((3, 3))
    t_best = np.zeros((3, 1))
    I = np.eye(3)
    P1 = np.eye(3, 4)
    pts_front_best = 0
    for R in Rs:
        for s2 in [1, -1]:
            t = s2*v
            P2 = R @ np.c_[I, -t]
            X = triangulate(u1, u2, P1, P2)
            pts_front = np.sum(X[2] > 0)
            print(f"got {pts_front}/{u1.shape[1]} points in front of both cameras")
            if pts_front > pts_front_best:
                # print(f"got {pts_front}/{u1.shape[1]} points in front of both cameras")
                pts_front_best = pts_front
                R_best = R
                t_best = t
    C_best = t_best * C_norm

    return R_best, C_best


def triangulate(u1, u2, P1, P2):
    n_points = u1.shape[1]
    o = np.zeros((3, 1))
    X = np.zeros((3, n_points))
    for i in range(n_points):
        u1_i = e2p(np.c_[u1[:, i]])
        u2_i = e2p(np.c_[u2[:, i]])

        # first approach
        A = np.r_[np.c_[u1_i,    o, -P1],
                  np.c_[   o, u2_i, -P2]]

        [U, S, Vt] = np.linalg.svd(A)
        Xi = Vt[-1, 2:] / Vt[-1, 5]       # last V column: [l1, l2, x1, x2, x3, x4]
        X[:, i] = Xi[:3]

        # second approach
        # still precise, but less points in front of camera
        A = np.r_[[P1[2,:]*u1_i[0]-P1[0,:]],
                  [P1[2,:]*u1_i[1]-P1[1,:]],
                  [P2[2,:]*u2_i[0]-P2[0,:]],
                  [P2[2,:]*u2_i[1]-P2[1,:]]]
        [U, S, Vt2] = np.linalg.svd(A)
        Xi2 = Vt2[-1,:] / Vt2[-1, 3]
        # X[:, i] = Xi2[:3]

    return X


if __name__ == "__main__":
    img1 = mpimg.imread("data/daliborka_01.jpg")
    img2 = mpimg.imread("data/daliborka_23.jpg")
    data_hw9a = sio.loadmat("data/09a_data.mat")

    [u1, u2, points_sel, Fe, E] = get_from_dict(data_hw9a, ('u1', 'u2', 'point_sel_e', 'Fe', 'E'))
    K = np.array(sio.loadmat("data/K.mat")["K"])

    R, C = E2RC(u1, u2, E)

    P1 = K @ np.eye(3, 4)
    P2 = K @ np.hstack([R, -R @ np.c_[C]])

    X = triangulate(u1, u2, P1, P2)

    u1_proj = p2e(P1 @ e2p(X))
    u2_proj = p2e(P2 @ e2p(X))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img1)
    ax2.imshow(img2)
    fig.suptitle("The reprojections", y=0.85, fontsize=14, fontweight="bold")

    ax1.set_title("Image 1")
    ax1.scatter(u1[0], u1[1], marker="o", c="b")
    ax1.scatter(u1_proj[0], u1_proj[1], marker="o", c="r")

    ax2.set_title("Image 2")
    ax2.scatter(u2[0], u2[1], marker="o", c="b")
    ax2.scatter(u2_proj[0], u2_proj[1], marker="o", c="r")

    plt.show()

    sio.savemat('09b_data.mat', {
        'Fe': Fe,
        'E': E,
        'R': R,
        'C': C,
        'P1': P1,
        'P2': P2,
        'X': X,
        'u1': u1,
        'u2': u1,
        'point_sel_e': points_sel,
    })
