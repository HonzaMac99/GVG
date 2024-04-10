import numpy as np                # for matrix computation and linear algebra
import matplotlib.pyplot as plt   # for drawing and image I/O
from mpl_toolkits.mplot3d import Axes3D  # for 3d plotting
import matplotlib.image as mpimg
# from PIL import Image
import scipy.io as sio            # for matlab file format output
import itertools                  # for generating all combinations

PRINT = False

IX = [86, 77, 83, 7, 20, 45, 63, 74, 26, 38]  # !!!matlab 1-based


def e2p(u):
    if len(u.shape) < 2:
        u = np.vstack(u)  # correct 1d arrays to be vertical

    n = u.shape[1]
    return np.vstack((u, np.ones((1, n))))


def p2e(p):
    if len(p.shape) < 2:
        p = np.vstack(p)  # correct 1d arrays to be vertical

    d = p.shape[0]

    assert d >= 3, "invalid dimension"
    assert p[d-1, :].all(), "division by zero"

    return p[:d-1, :] / p[d-1, :]


# compute Q from the 6 u-X correspondences and remove i-th column
def compute_Q(u, x, i_remove):
    M = np.zeros((12, 12))
    for i in range(6):
        M[i*2, :]   = np.hstack((e2p(x[:, i]).flatten(),             np.zeros(4),  -u[0, i]*e2p(x[:, i]).flatten()))
        M[i*2+1, :] = np.hstack((           np.zeros(4),  e2p(x[:, i]).flatten(),  -u[1, i]*e2p(x[:, i]).flatten()))

    # remove one row of M
    M = np.delete(M, i_remove, 0)

    # check for solvability
    assert np.linalg.matrix_rank(M) >= min(M.shape), "Columns of M are not linearly independent."

    # solve Mq = 0
    U, S, Vt = np.linalg.svd(M)
    q = Vt[-1, :]  # last row

    # q normalisation - doesnt seem to have effect here
    # q = q / q[-1]

    # Q = np.vstack((q[:4].T, q[4:8].T, q[8:].T))
    Q = q.reshape(3, 4)
    return Q


def estimate_Q(u_all, x_all, ix):
    points_sel = np.zeros(ix.shape[0])
    err_max = []
    err_points = np.zeros((u_all.shape[1]))
    Q_all = []

    iter = itertools.combinations(range(0, 10), 6)  # every 6 pts of 10 pts

    e_max_best = np.inf
    Q_best = np.zeros((3, 3))
    for inx in iter:
        ix_sel = ix[np.array(inx)]  # corresp. indexes
        u_sel = u_all[:, ix_sel]
        x_sel = x_all[:, ix_sel]

        for i in range(12):
            Q = compute_Q(u_sel, x_sel, i)

            u_proj = p2e(Q @ e2p(x_all))
            e = np.sqrt(np.power(u_all[0]-u_proj[0], 2) + np.power(u_all[1]-u_proj[1], 2))
            e_max = np.max(e)

            Q_all.append(Q)
            err_max.append(e_max)

            if e_max < e_max_best:
                if PRINT:
                    print("New best e:", e_max)
                e_max_best = e_max
                Q_best = Q
                points_sel = ix_sel
                err_points = e

    return Q_best, points_sel, err_max, err_points, Q_all


if __name__ == "__main__":
    img_arr = mpimg.imread("daliborka_01.jpg")
    ux_all = sio.loadmat("daliborka_01-ux.mat")
    u_all, x_all = ux_all['u'], ux_all['x']
    ix = np.array(IX) - 1  # convert from matlab to python indexing
    Q, points_sel, err_max, err_points, Q_all = estimate_Q(ux_all['u'], ux_all['x'], ix)

    # # plot the 3d points
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_all[0, :], x_all[1, :], x_all[2, :], c='b', marker='o', s=10)
    # plt.show()

    fig = plt.figure()

    # draw all points and their reprojections
    plt.imshow(img_arr)
    u_proj = p2e(Q @ e2p(x_all))
    u_sel = u_all[:, points_sel]
    for i in range(u_all.shape[1]):
        if i == points_sel[0]:
            plt.plot(u_all[0, i], u_all[1, i], 'o', color='y', fillstyle='full', mec="k", mew=0.5, label='Used for Q')
        elif i in points_sel[0:]:
            plt.plot(u_all[0, i], u_all[1, i], 'o', color='y', fillstyle='full', mec="k", mew=0.5)
        elif i == 0:
            plt.plot(u_all[0, i], u_all[1, i], 'o', color='b', markersize=2, label='Orig. points')
            plt.plot(u_proj[0, i], u_proj[1, i], 'o', color='r', fillstyle='none', label='Reprojected')
        else:
            plt.plot(u_all[0, i], u_all[1, i], 'o', color='b', markersize=2)
        plt.plot(u_proj[0, i], u_proj[1, i], 'o', color='r', fillstyle='none')

    plt.title("Original and reprojected points")
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.legend(loc='lower right')
    plt.show()
    fig.savefig('02_Q_projections.pdf')
    fig = plt.figure()


    # draw all points and their reprj. errors
    plt.imshow(img_arr)
    e = 100 * (u_proj - u_all)
    for i in range(u_all.shape[1]):
        if i == points_sel[0]:
            plt.plot(u_all[0, i], u_all[1, i], 'bo', color='y', fillstyle='full', mec="k", mew=0.5, label='Used for Q')
        elif i in points_sel[0:]:
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
    fig.savefig('02_Q_projections_errors.pdf')
    fig = plt.figure()

    plt.title("Maximal reproj. err. for each tested Q")
    plt.plot(np.log10(np.array(err_max)))
    plt.xlabel("selection index")
    plt.ylabel("log10 of max reproj. err. [px]")
    plt.show()
    fig.savefig('02_Q_maxerr.pdf')
    fig = plt.figure()

    plt.title("All point reproj. errors for the best Q")
    plt.plot(err_points)
    plt.xlabel("point index")
    plt.ylabel("reproj. err. [px]")
    plt.show()
    fig.savefig('02_Q_pointerr.pdf')

    # save the Q for hw03
    np.save("Q.npy", Q)
    np.save("Q_all.npy", Q_all)








