import numpy as np                # for matrix computation and linear algebra
import matplotlib.pyplot as plt   # for drawing and image I/O
import scipy
from scipy import linalg
from mpl_toolkits import mplot3d
import matplotlib.image as mpimg
# from PIL import Image
import scipy.io as sio            # for matlab file format output
import itertools                  # for generating all combinations


# b1_len = 5e-6  # [m]
b1_len = 5e-4  # [m]


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


def Q2KRC(Q):
    K = np.eye(3)
    R = np.eye(3)
    C = np.zeros((3, 1))

    # # f = 1 / Q[2, 2]
    # K[1, 1] = np.sqrt(np.power(Q[1, 0], 2) + np.power(Q[1, 1], 2))
    # s = Q[1, 0] / K[1, 1]
    # c = Q[1, 1] / K[1, 1]

    # K[0, 2] = Q[0, 2]
    # K[1, 2] = Q[1, 2]

    # R[0:2, 0:2] = np.array([[c, -s], [s, c]])

    # # K[0, 0] =
    # # K[0, 1] =

    K, R = scipy.linalg.rq(Q[:, :3])

    S = np.diag([-1, 1, 1])

    if K[2, 2] < 0:
        K *= -1
        R *= -1

    if K[0, 0] < 0:
        K = K @ S
        R = S @ R

    C = - np.linalg.solve(Q[:, :3], Q[:, 3]).reshape(3, 1)

    return [K, R, C]


def plot_csystem(ax, Base = np.eye(3), b = np.zeros((3, 1)), color = 'k', name = 'd'):
    n_axes = 3 if (name != 'a') else 2
    multipliers = [1, 1, 1]
    if name in ['a', 'b']:
        multipliers = [1100, 850, 1]

    b = b.reshape(3,)
    for i in range(n_axes):
        x_crds = [b[0], b[0] + Base[0, i]*multipliers[i]]
        y_crds = [b[1], b[1] + Base[1, i]*multipliers[i]]
        z_crds = [b[2], b[2] + Base[2, i]*multipliers[i]]
        ax.plot(x_crds, y_crds, z_crds, c=color, marker="^")
        ax.text(x_crds[1], y_crds[1], z_crds[1], f"{name}{i+1}", c=color, fontsize=10)
    return


def fix_axes(ax, n):
    ax.set_xlim3d(-n, n)
    ax.set_ylim3d(-n, n)
    ax.set_zlim3d(-n, n)
    return


if __name__ == "__main__":
    ux_all = sio.loadmat("daliborka_01-ux.mat")
    u_all, x_all = ux_all['u'], ux_all['x']
    Q = np.load("Q.npy")
    Q /= np.linalg.norm(Q[2, :3])

    [K, R, C] = Q2KRC(Q)
    f = K[0, 0] * b1_len

    # print(f"K = \n{K}")
    # print(f"R = \n{R}")
    # print(f"C = \n{C}")
    # print("-------------------")
    Pb = (1/f) * np.hstack((K@R, -K@R@C))
    # print(f"Pb = \n{Pb}")

    R_inv = R.T
    K_inv = np.linalg.inv(K)
    O = np.zeros((3, 1))
    b3_beta = np.array([0, 0, 1]).reshape(3, 1)
    b3_delta = f*R_inv@K_inv@b3_beta
    o = C + b3_delta

    Delta,   d = np.eye(3), O
    Kappa,   k = Delta * f, O
    Epsilon, e = Delta @ R_inv, C
    Gamma,   g = Epsilon * f, C
    Nu,      n = Epsilon @ K_inv, C
    Beta,    b = Gamma @ K_inv, C
    Alpha,   a = Beta[:, 0:2], o

    sio.savemat('03_bases.mat', {'Pb': Pb, 'f': f,
                                 'Alpha': Alpha, 'a': a,
                                 'Beta': Beta, 'b': b,
                                 'Gamma': Gamma, 'g': g,
                                 'Delta': Delta, 'd': d,
                                 'Epsilon': Epsilon, 'e': e,
                                 'Kappa': Kappa, 'k': k,
                                 'Nu': Nu, 'n': n})

    params = {"Alpha"   : [Alpha,   a, 'g',     'a'],
              "Beta"    : [Beta,    b, 'r',     'b'],
              "Gamma"   : [Gamma,   g, 'b',     'c'],
              "Delta"   : [Delta,   d, 'k',     'd'],
              "Epsilon" : [Epsilon, e, 'm',     'e'],
              "Kappa"   : [Kappa,   k, "brown", 'k'],
              "Nu"      : [Nu,      n, 'c',     'n']}

    bases1 = ["Delta", "Epsilon", "Kappa", "Nu", "Beta"]
    bases2 = ["Alpha", "Beta", "Gamma"]
    # bases2 = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    bases3 = ["Delta", "Epsilon"]

    fig = plt.figure(figsize = (8,8))
    ax = plt.axes(projection='3d')
    fix_axes(ax, 2)
    for i in range(len(bases1)):
        ps = params[bases1[i]]
        plot_csystem(ax, ps[0], ps[1], ps[2], ps[3])
    # TODO: why the img points are 2*times further than the image coords?
    u_all_delta = (Beta @ e2p(u_all)) + o   # Beta = R_inv * f @ K_inv
    ax.scatter(u_all_delta[0, :], u_all_delta[1, :], u_all_delta[2, :], marker='o', s=20, c='b')
    plt.show()

    fig = plt.figure(figsize = (8,8))
    ax = plt.axes(projection='3d')
    fix_axes(ax, 2)
    for i in range(len(bases2)):
        ps = params[bases2[i]]
        plot_csystem(ax, ps[0], ps[1], ps[2], ps[3])
    # TODO: why the img points are 2*times further than the image coords?
    u_all_delta = (Beta @ e2p(u_all)) + o   # Beta = R_inv * f @ K_inv
    ax.scatter(u_all_delta[0, :], u_all_delta[1, :], u_all_delta[2, :], marker='o', s=20, c='b')
    # ax.scatter(x_all[0, :], x_all[1, :], x_all[2, :], marker='o', s=20, c='r')
    plt.show()

    fig = plt.figure(figsize = (8, 8))
    ax = plt.axes(projection='3d')
    fix_axes(ax, 2)
    for i in range(len(bases3)):
        ps = params[bases3[i]]
        plot_csystem(ax, ps[0], ps[1], ps[2], ps[3])
    ax.scatter(x_all[0, :], x_all[1, :], x_all[2, :], marker='o', s=20, c='b')
    plt.show()

    # Set axes label
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    ax.set_title('Coordinate systems')

    plt.show()

