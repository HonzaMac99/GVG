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


def get_err(x, y, d, c):
    e = (np.sqrt(x**2 + y**2 - 2*x*y*c) - d) / d
    return e


def p3p_dverify(n1, n2, n3, d12, d23, d31, c12, c23, c31):

    e1 = get_err(n1, n2, d12, c12)
    e2 = get_err(n2, n3, d23, c23)
    e3 = get_err(n1, n3, d31, c31)
    return np.array([e1, e2, e3])


def get_ai_coeff(d12, d23, d31, c12, c23, c31):
    a4 = - (4 * d23**4 * d12**2 * d31**2 * c23**2) + (d23**8) - (2 * d23**6 * d12**2) - (2 * d23**6*d31**2)\
         + (d23**4 * d12**4) + (2 * d23**4 * d12**2 * d31**2) + (d23**4 * d31**4)

    a3 = (8 * d23**4 * d12**2 * d31**2 * c12 * c23**2) + (4 * d23**6 * d12**2 * c31 * c23)\
         - (4 * d23**4 * d12**4 * c31 * c23) + (4 * d23**4 * d12**2 * d31**2 * c31 * c23) - (4 * d23**8 * c12)\
         + (4 * d23**6 * d12**2 * c12) + (8 * d23**6 * d31**2 * c12) - (4 * d23**4 * d12**2 * d31**2 * c12)\
         - (4 * d23**4 * d31**4 * c12)

    a2 = - (8 * d23**6 * d12**2 * c31 * c12 * c23) - (8 * d23**4 * d12**2 * d31**2 * c31 * c12 * c23) + (4 * d23**8 * c12**2)\
         - (4 * d23**6 * d12**2 * c31**2) - (8 * d23**6 * d31**2 * c12**2) + (4 * d23**4 * d12**4 * c31**2)\
         + (4 * d23**4 * d12**4 * c23**2) - (4 * d23**4 * d12**2 * d31**2 * c23**2) + (4 * d23**4 * d31**4 * c12**2)\
         + (2 * d23**8) - (4 * d23**6 * d31**2) - (2 * d23**4 * d12**4) + (2 * d23**4 * d31**4)

    a1 = (8 * d23**6 * d12**2 * c31**2 * c12) + (4 * d23**6 * d12**2 * c31 * c23) - (4 * d23**4 * d12**4 * c31 * c23)\
         + (4 * d23**4 * d12**2 * d31**2 * c31 * c23) -(4 * d23**8 * c12) - (4 * d23**6 * d12**2 * c12)\
         + (8 * d23**6 * d31**2 * c12) + (4 * d23**4 * d12**2 * d31**2 * c12) - (4 * d23**4 * d31**4 * c12)

    a0 = - (4 * d23**6 * d12**2 * c31**2) + (d23**8 - 2 * d23**4 * d12**2 * d31**2) + (2 * d23**6 * d12**2)\
         + (d23**4 * d31**4) + (d23**4 * d12**4) - (2 * d23**6 * d31**2)

    return a0, a1, a2, a3, a4


def p3p_distances(d12, d23, d31, c12, c23, c31):
    '''
    Computation of distances of three spatial points from a center of calibrated camera
    Only the case 'A' is implemented
    Input: d12, d23, d31 ... distances between the spatial points
           c12, c23, c31 ... cosines of angles between the projection rays
    '''
    n1s, n2s, n3s = [], [], []
    th = 1e-4

    # get polynomial coeffs from eqs 6.82 - 6.86
    a0, a1, a2, a3, a4 = get_ai_coeff(d12, d23, d31, c12, c23, c31)

    # construct the companion matrix
    C = np.array([
        [0, 0, 0, -a0 / a4],
        [1, 0, 0, -a1 / a4],
        [0, 1, 0, -a2 / a4],
        [0, 0, 1, -a3 / a4]
    ])

    # compute its eigenvalues
    n12s = np.linalg.eigvals(C)

    for n12 in n12s:
        if np.iscomplex(n12):  # Complex solutions should not be considered
            continue

        n12 = n12.real

        # get params from eqs. 7.69 - 7.74 (or slides eq 6.69 - 6.74)
        m1 = d12**2
        p1 = -2 * d12**2 * n12 * c23
        q1 = d23**2 * (1 + n12**2 - 2 * n12 * c12) - d12**2 * n12**2
        m2 = d31**2 - d23**2
        p2 = 2 * d23**2 * c31 - 2 * d31**2 * n12 * c23
        q2 = d23**2 - d31**2 * n12**2

        # 7.89
        n13 = (m1 * q2 - m2 * q1) / (m1 * p2 - m2 * p1)

        # 7.91 - 7.93
        n1 = d12 / np.sqrt(1 + n12**2 - 2 * n12 * c12)
        n2 = n1 * n12
        n3 = n1 * n13

        errs = p3p_dverify(n1, n2, n3, d12, d23, d31, c12, c23, c31)

        if np.max(errs) < th:
            n1s.append(n1)
            n2s.append(n2)
            n3s.append(n3)

    return n1s, n2s, n3s


def get_dists(x1, x2, x3):
    d12 = np.linalg.norm(x1 - x2)
    d23 = np.linalg.norm(x2 - x3)
    d31 = np.linalg.norm(x1 - x3)

    return d12, d23, d31


def get_cosines(x1, x2, x3, K):
    K_inv = np.linalg.inv(K)

    norm1 = np.linalg.norm(K_inv@x1)
    norm2 = np.linalg.norm(K_inv@x2)
    norm3 = np.linalg.norm(K_inv@x3)

    c12 = float(x1.T@K_inv.T@K_inv@x2) / (norm1 * norm2)
    c23 = float(x2.T@K_inv.T@K_inv@x3) / (norm2 * norm3)
    c31 = float(x3.T@K_inv.T@K_inv@x1) / (norm3 * norm1)

    return c12, c23, c31


if __name__ == "__main__":

    # --- Task 1.1 ---
    I = np.eye(3)
    K, R = I, I

    C = np.array([1, 2, -3]).reshape(3, 1)
    f = 1

    P = (1/f) * K @ R @ np.hstack((I, -C))

    X1 = np.array([0, 0, 0]).reshape(3, 1)
    X2 = np.array([1, 0, 0]).reshape(3, 1)
    X3 = np.array([0, 1, 0]).reshape(3, 1)

    # project points by P
    x1 = P @ e2p(X1)
    x2 = P @ e2p(X2)
    x3 = P @ e2p(X3)

    # normalize the points
    x1 = e2p(p2e(x1))
    x2 = e2p(p2e(x2))
    x3 = e2p(p2e(x3))

    c12, c23, c31 = get_cosines(x1, x2, x3, K)
    d12, d23, d31 = get_dists(X1, X2, X3)

    # get the dists between camera centre and the points
    n1s, n2s, n3s = p3p_distances(d12, d23, d31, c12, c23, c31)

    # get the ground truth dists for comparison
    n1_gt = np.linalg.norm(C - X1)
    n2_gt = np.linalg.norm(C - X2)
    n3_gt = np.linalg.norm(C - X3)

    print("--- Task 1.1 ---")
    print(f"gt: {n1_gt},   my_solution: {n1s}")
    print(f"gt: {n2_gt},   my_solution: {n2s}")
    print(f"gt: {n3_gt},   my_solution: {n3s}\n")


    # --- Task 1.2 ---
    c12 = 0.9037378393
    c23 = 0.8269612542
    c31 = 0.9090648231

    X1 = np.array([1, 0, 0]).reshape(3, 1)
    X2 = np.array([0, 2, 0]).reshape(3, 1)
    X3 = np.array([0, 0, 3]).reshape(3, 1)

    d12, d23, d31 = get_dists(X1, X2, X3)

    # get the dists between camera centre and the points
    n1s, n2s, n3s = p3p_distances(d12, d23, d31, c12, c23, c31)

    print("--- Task 1.2 ---")
    print(n1s)
    print(n2s)
    print(n3s)


    # --- Task 2 ---
    K = sio.loadmat("K.mat")["K"]
    ux_all = sio.loadmat("daliborka_01-ux.mat")
    u_all, x_all = ux_all['u'], ux_all['x']

    ix = np.array([86, 77, 83, 7, 20, 45, 63, 74, 26, 38]) - 1

    iter = itertools.combinations(range(0, len(ix)), 3)  # every 3 pts of 10 pts

    n1s_all, n2s_all, n3s_all = [], [], []
    for inx in iter:
        ix_sel = ix[np.array(inx)]

        X1, X2, X3 = x_all[:, ix_sel].T
        x1, x2, x3 = e2p(u_all[:, ix_sel]).T
        c12, c23, c31 = get_cosines(x1, x2, x3, K)
        d12, d23, d31 = get_dists(X1, X2, X3)
        n1s, n2s, n3s = p3p_distances(d12, d23, d31, c12, c23, c31)

        n1s_all.extend(n1s)
        n2s_all.extend(n2s)
        n3s_all.extend(n3s)

    # plot the results
    plt.plot(n1s_all, color='red', label="$\\eta_1$")
    plt.plot(n2s_all, color='blue', label="$\\eta_2$")
    plt.plot(n3s_all, color='green', label="$\\eta_3$")

    plt.xlim(0, 250)
    plt.ylim(0, 0.9)

    plt.title('Point distances')
    plt.ylabel('distance [m]')
    plt.xlabel('trial')

    plt.legend()
    plt.savefig('04_distances.pdf')
    plt.show()

