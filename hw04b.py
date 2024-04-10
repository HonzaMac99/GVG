import numpy as np                # for matrix computation and linear algebra
import matplotlib.pyplot as plt   # for drawing and image I/O
import scipy
from scipy import linalg
from mpl_toolkits import mplot3d
import matplotlib.image as mpimg
# from PIL import Image
import scipy.io as sio            # for matlab file format output
import itertools                  # for generating all combinations

from hw03 import *
from hw04a import *

IX = [86, 77, 83, 7, 20, 45, 63, 74, 26, 38]  # !!!matlab 1-based


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


def p3p_RC(N, u, X, K):
    [n1, n2, n3] = N
    K_inv = np.linalg.inv(K)
    x1, x2, x3 = np.hsplit((K_inv @ e2p(u)), 3)
    # u1, u2, u3 = np.hsplit(e2p(u), 3)
    X1, X2, X3 = np.hsplit(X, 3)

    # 3D points in camera coords
    Y1 = n1 * x1 / np.linalg.norm(x1)
    Y2 = n2 * x2 / np.linalg.norm(x2)
    Y3 = n3 * x3 / np.linalg.norm(x3)

    Z2e = Y2 - Y1
    Z3e = Y3 - Y1
    Z1e = np.cross(Z2e.flatten(), Z3e.flatten()).reshape(3, 1)
    Ze = np.hstack([Z1e, Z2e, Z3e])

    Z2d = X2 - X1
    Z3d = X3 - X1
    Z1d = np.cross(Z2d.flatten(), Z3d.flatten()).reshape(3, 1)
    Zd = np.hstack([Z1d, Z2d, Z3d])

    R = Ze @ np.linalg.inv(Zd)

    # i = 1, 2, 3 doesn't matter which idx
    C = X1 - R.T@Y1

    return R, C


if __name__ == "__main__":

    # ------------------ debuging p3p_RC ------------------
    K = R = I = np.eye(3)
    X1 = np.array([0, 0, 0]).reshape(3, 1)
    X2 = np.array([1, 0, 0]).reshape(3, 1)
    X3 = np.array([0, 1, 0]).reshape(3, 1)
    X = np.hstack([X1, X2, X3])

    C = np.array([1, 2, -3]).reshape(3, 1)
    f = 1

    P = (1/f) * K @ R @ np.hstack((I, -C))

    x = P @ e2p(X)
    x = e2p(p2e(x))
    x1, x2, x3 = np.hsplit(x, 3)

    c12, c23, c31 = get_cosines(x1, x2, x3, K)
    d12, d23, d31 = get_dists(X1, X2, X3)
    n1s, n2s, n3s = p3p_distances(d12, d23, d31, c12, c23, c31)
    N = [n1s[0], n2s[0], n3s[0]]

    u = p2e(x)
    R_new, C_new = p3p_RC(N, u, X, K)

    print(R_new)
    print(C_new)

    # -----------------------------------------------------

    img_arr = mpimg.imread("daliborka_01.jpg")
    K = sio.loadmat("K.mat")["K"]
    ux_all = sio.loadmat("daliborka_01-ux.mat")
    u_all, x_all = ux_all['u'], ux_all['x']
    ix = np.array(IX) - 1  # convert from matlab to python indexing

    C_all = []
    err_max = []
    e_max_best = np.inf
    err_points = np.zeros((u_all.shape[1]))
    R_best, C_best, ix_sel_best = None, None, None
    for inx in itertools.combinations(range(0, 10), 3):  # three of n
        ix_sel = ix[np.array(inx)]  # corresp. indexes
        u_i = u_all[:, ix_sel]
        X_i = x_all[:, ix_sel]

        K_inv = np.linalg.inv(K)
        # x = K_inv @ e2p(u_i)
        # x = e2p(p2e(x))
        # x1, x2, x3 = np.hsplit(x, 3)
        x1, x2, x3 = np.hsplit(e2p(u_i), 3)
        X1, X2, X3 = np.hsplit(X_i, 3)

        c12, c23, c31 = get_cosines(x1, x2, x3, K)
        d12, d23, d31 = get_dists(X1, X2, X3)
        n1s, n2s, n3s = p3p_distances(d12, d23, d31, c12, c23, c31)

        for i in range(len(n1s)):
            N = [n1s[i], n2s[i], n3s[i]]

            R_new, C_new = p3p_RC(N, u_i, X_i, K)
            Q_new = K @ R_new @ np.hstack((I, -C_new))

            u_proj = p2e(Q_new @ e2p(x_all))
            e = np.sqrt(np.power(u_all[0]-u_proj[0], 2) + np.power(u_all[1]-u_proj[1], 2))
            e_max = np.max(e)

            C_all.append(C_new)
            err_max.append(e_max)

            if e_max < e_max_best:
                print("New best e:", e_max)
                e_max_best = e_max
                R_best, C_best, ix_sel_best = R_new, C_new, ix_sel
                err_points = e

    print(R_new)
    print(C_new)



    # 1) draw all points and their reprj. errors
    fig = plt.figure()
    plt.imshow(img_arr)
    Q = K @ R_best @ np.hstack((I, -C_best))
    u_proj = p2e(Q @ e2p(x_all))
    e = 100 * (u_proj - u_all)
    for i in range(u_all.shape[1]):
        if i == ix_sel_best[0]:
            plt.plot(u_all[0, i], u_all[1, i], 'bo', color='y', fillstyle='full', mec="k", mew=0.5, label='Used for Q')
        elif i in ix_sel_best[0:]:
            plt.plot(u_all[0, i], u_all[1, i], 'bo', color='y', fillstyle='full', mec="k", mew=0.5)
        elif i == 0:
            plt.plot(u_all[0, i], u_all[1, i], 'o', color='b', markersize=1, label='Orig. points')
            plt.plot((u_all[0, i], u_all[0, i] + e[0, i]), (u_all[1, i], u_all[1, i] + e[1, i]), 'r-', label='Errors (100x)')
        else:
            plt.plot(u_all[0, i], u_all[1, i], 'o', color='b', markersize=1)
        plt.plot((u_all[0, i], u_all[0, i] + e[0, i]), (u_all[1, i], u_all[1, i] + e[1, i]), 'r-')

    plt.title("Reprojection errors (100x enlarged)")
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.legend(loc='lower right')
    plt.show()
    fig.savefig('04_RC_projections_errors.pdf')
    fig = plt.figure()

    # 2) plot the max. reprj. errors for every solution
    plt.title("Maximal reproj. err. for each tested P")
    plt.plot(np.log10(np.array(err_max)), "bo", markersize=2.0)
    plt.ylim(0, 4.5)
    plt.xlabel("selection index")
    plt.ylabel("log10 of max reproj. err. [px]")
    plt.show()
    fig.savefig('04_RC_maxerr.pdf')
    fig = plt.figure()

    # 3) plot errors for the best solution
    plt.title("All point reproj. errors for the best P")
    plt.plot(err_points)
    plt.xlabel("point index")
    plt.ylabel("reproj. err. [px]")
    plt.show()
    fig.savefig('04_RC_pointerr.pdf')

    # 4) plot all extimations of C
    Delta,   d = np.eye(3),         np.zeros((3, 1))
    Epsilon, e = Delta @ R_best.T,  C_best

    params = [[Delta,   d, 'k',     "delta"],
              [Epsilon, e, 'm',     "epsilon"]]

    fig = plt.figure(figsize = (8, 8))
    ax = plt.axes(projection='3d')
    fix_axes(ax, 1)
    # plot the centres of all Qs
    for C in C_all:
        ax.scatter(C[0, :], C[1, :], C[2, :], marker='o', s=4, c='r')  #, alpha=0.25)

    for i in range(len(params)):
        ps = params[i]
        plot_csystem(ax, ps[0], ps[1], ps[2], ps[3])
    ax.scatter(x_all[0, :], x_all[1, :], x_all[2, :], marker='o', s=20, c='b')  #, alpha=0.5)

    ax.set_xlabel('x [m]', labelpad=20)
    ax.set_ylabel('y [m]', labelpad=20)
    ax.set_zlabel('z [m]', labelpad=20)
    ax.set_title('Estimated camera centres')
    plt.show()
    fig.savefig('04_scene.pdf')


    sio.savemat('04_p3p.mat', {'R': R_best, 'C': C_best, 'point_sel': ix_sel_best})



