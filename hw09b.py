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
    Rs = np.empty(2, dtype=object)
    for i, s in enumerate([1, -1]):
        # first approach (eq. 12.63) --> we get R, that is not orthonormal (R.T@R != 0 and det(R) != 1)!
        A = np.c_[s*g1, s*g2, s*g3, np.cross(g1, g2), np.cross(g2, g3), np.cross(g3, g1)]
        B = np.c_[  v1,   v2,   v3, np.cross(v1, v2), np.cross(v2, v3), np.cross(v3, v1)]

        # A = R@B --> B.T@R.T = A.T
        R = (np.linalg.lstsq(B.T, A.T, rcond=None)[0]).T
        # R = np.linalg.solve(B.T, A.T).T  # only for square matrices
        # Rs[i] = R

        # second approach (eq. 12.103)
        alpha = s
        W = np.array([[     0, alpha, 0],
                      [-alpha,     0, 0],
                      [     0,     0, 1]])
        R = U@W@Vt
        Rs[i] = R

    R_best = np.zeros((3, 3))
    t_best = np.zeros((3, 1))
    I = np.eye(3)
    P1 = np.eye(3, 4)
    pts_front_best = 0
    for R in Rs:
        for s2 in [1, -1]:
            t = -s2*v
            P2 = R @ np.c_[I, -t]
            X = triangulate(u1, u2, P1, P2)
            u1_proj = P1@e2p(X)
            u2_proj = P2@e2p(X)
            pts_front = np.sum((u1_proj[2] > 0) * (u2_proj[2] > 0))
            if pts_front > pts_front_best:
                print(f"got {pts_front}/{u1.shape[1]} points in front of both cameras")
                pts_front_best = pts_front
                R_best = R
                t_best = np.c_[t]
    C_best = t_best * C_norm

    return R_best, C_best


def triangulate(u1, u2, P1, P2):
    n_points = u1.shape[1]
    o = np.zeros((3, 1))
    X = np.zeros((3, n_points))
    for i in range(n_points):
        u1_i = e2p(np.c_[u1[:, i]])
        u2_i = e2p(np.c_[u2[:, i]])

        # first approach (eq. 12.69) - gives wrong results
        A = np.r_[np.c_[u1_i,    o, -P1],
                  np.c_[   o, u2_i, -P2]]

        [U, S, Vt] = np.linalg.svd(A)
        Xi = Vt[-1, 2:5] / Vt[-1, 5]       # last V column: [lambda1, lambda2, x1, x2, x3, x4]

        # second approach
        # better reprj. error, but less points in front of camera
        A = np.r_[[P1[2,:]*u1_i[0] - P1[0,:]],
                  [P1[2,:]*u1_i[1] - P1[1,:]],
                  [P2[2,:]*u2_i[0] - P2[0,:]],
                  [P2[2,:]*u2_i[1] - P2[1,:]]]
        [U, S, Vt2] = np.linalg.svd(A)
        Xi2 = Vt2[-1, :3] / Vt2[-1, 3]

        # X[:, i] = Xi
        X[:, i] = Xi2

    return X


if __name__ == "__main__":
    img1 = mpimg.imread("data/daliborka_01.jpg")
    img2 = mpimg.imread("data/daliborka_23.jpg")
    edges = sio.loadmat("data/daliborka_01_23-uu.mat")['edges'] - 1   # convert matlab indexing format
    data_hw9a = sio.loadmat("data/09a_data.mat")

    [u1, u2, points_sel, Fe, E] = get_from_dict(data_hw9a, ('u1', 'u2', 'point_sel_e', 'Fe', 'E'))
    K = np.array(sio.loadmat("data/K.mat")["K"])

    # === step 1 ===
    R, C = E2RC(u1, u2, E)

    # === step 2 ===
    P1 = K @ np.eye(3, 4)
    P2 = K @ np.hstack([R, -R @ np.c_[C]])

    # === step 3 ===
    X = triangulate(u1, u2, P1, P2)

    # === step 4 ===
    u1_proj = p2e(P1 @ e2p(X))
    u2_proj = p2e(P2 @ e2p(X))
    plot_reprojections(img1, img2, u1, u2, u1_proj, u2_proj, edges)

    # === step 5 ===
    d1 = np.linalg.norm(u1 - u1_proj, axis=0)
    d2 = np.linalg.norm(u2 - u2_proj, axis=0)
    n_points = u1.shape[1]

    plot_rp_errors([d1, d2], "09_errorsr.pdf")

    # === step 6 ===
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-2.5, -0.5])
    # ax.view_init(elev=90, azim=-90, roll=0)
    # ax.axis("off")

    ax.scatter(X[0], X[1], X[2], marker="o", s=2, color="b")
    ax.scatter(C[0], C[1], C[2], marker="o", s=5, color="r")
    ax.text(C[0, 0], C[1, 0], C[2, 0], "Camera", c="r", fontsize=6)

    # plot the edges on bottom
    for i in range(edges.shape[1]):
        e = edges[:, i]
        x1, y1, z1 = X[:, e[0]]
        x2, y2, z2 = X[:, e[1]]
        ax.plot([x1, x2], [y1, y2], [z1, z2], "-", color="royalblue", zorder=-1)

    plt.show()

    # === step 7 ===
    sio.savemat('09b_data.mat', {
        'Fe': Fe,
        'E': E,
        'R': R,
        'C': C,
        'P1': P1,
        'P2': P2,
        'X': X,
        'u1': u1,
        'u2': u2,
        'point_sel_e': points_sel,
    })
