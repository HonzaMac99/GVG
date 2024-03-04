import numpy as np                # for matrix computation and linear algebra
import matplotlib.pyplot as plt   # for drawing and image I/O
import matplotlib.image as mpimg
# from PIL import Image
import scipy.io as sio            # for matlab file format output
import itertools                  # for generating all combinations

IX = [86, 77, 83, 7, 20, 45, 63, 74, 26, 38]  # !!!matlab 1-based

def e2p(u):
    n = u.shape[1]
    return np.vstack((u, np.ones((1, n))))


def Q2KRC(Q):
    K = np.eye(3)
    R = np.eye(3)
    C = np.zeros(3)

    f = 1 / Q[2, 2]
    K[1, 1] = np.sqrt(np.power(Q[1, 0], 2) + np.power(Q[1, 1], 2))
    s = Q[1, 0] / K[1, 1]
    c = Q[1, 1] / K[1, 1]

    K[0, 2] = Q[0, 2]
    K[1, 2] = Q[1, 2]

    R[0:2, 0:2] = np.array([[c, -s], [s, c]])

    # K[0, 0] =
    # K[0, 1] =


    return [K, R, C]


def plot_csystem(Base = np.eye(3), C = np.zeros(3, 1), color = 'k', name = 'd'):

    return


# compute Q from the 6 u-X correspondences and remove i-th column
def compute_Q(u, x, i_remove):
    b = np.zeros((12, 1))

    M = np.zeros((12, 12))
    for i in range(6):
        M[i*2, :]   = np.hstack((x[:, i].T, np.zeros(3), -u[0, i]*x[:, i].T))
        M[i*2+1, :] = np.hstack((np.zeros(3), x[:, i].T, -u[1, i]*x[:, i].T))

    # remove one row of M
    M = np.delete(M, i_remove, 0)

    # solve Ax = b
    q = np.linalg.solve(M, b)
    Q = np.vstack((q[:3].T, q[3:6].T, q[6:].T))
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
        ix_sel = ix[[inx]]
        u_sel = u_all[ix_sel]
        x_sel = x_all[ix_sel]

        for i in range(12):
            Q = compute_Q(u_sel, x_sel, i)

            u_proj = Q @ x_all
            e = np.power(u_all[0]-u_proj[0], 2) + np.power(u_all[1]-u_proj[1], 2)
            e_max = np.sqrt(np.max(e))

            Q_all.append(Q)
            err_max.append(e_max)

            if e_max < e_max_best:
                Q_best = Q
                points_sel = ix_sel
                err_points = e

    return Q_best, points_sel, err_max, err_points, Q_all


if __name__ == "__main__":
    img_arr = mpimg.imread("daliborka_01.jpg")
    ux_all = sio.loadmat("daliborka_01-ux.mat")
    ix = np.array(IX) - 1  # convert from matlab indexing to python
    Q, points_sel, err_max, err_points, Q_all = estimate_Q(ux_all['u'], ux_all['x'], ix)

    fig = plt.figure()  # figure handle to be used later
    fig.clf()
    plt.imshow(img_arr)

    # draw all points (in proper color) and errors
    u_all = ux_all['u']
    u_sel = u_all[points_sel]
    for i in range(u_all.shape[1]):
        plt.plot(u_all[0, i], u_all[1, i], 'o', color='b', fillstyle='none')
        plt.plot((u_all[0, i], u_all[0, i] + err_points[0, i]), (u_all[1, i], u_all[1, i] + err_points[1, i]), 'r-')

    plt.show()
    fig.savefig('01_daliborka_errs.pdf')








