import numpy as np                # for matrix computation and linear algebra
import matplotlib.pyplot as plt   # for drawing and image I/O
import scipy
from scipy import linalg
from mpl_toolkits import mplot3d
import matplotlib.image as mpimg
# from PIL import Image
import scipy.io as sio            # for matlab file format output
import itertools                  # for generating all combinations


b1_len = 5e-6  # [m]
# b1_len = 5e-4  # [m]


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
    lambd = np.linalg.norm(Q[2, :3])
    Q /= lambd

    K, R = scipy.linalg.rq(Q[:, :3])

    S = np.diag([-1, 1, 1])

    if K[2, 2] < 0:
        K *= -1
        R *= -1

    if K[0, 0] < 0:
        K = K @ S
        R = S @ R

    C = - np.linalg.solve(Q[:, :3], Q[:, 3]).reshape(3, 1)

    return K, R, C


def plot_csystem(ax, Base = np.eye(3), b = np.zeros((3, 1)), color = 'k', name = 'd', scale=1):
    n_axes = 3 if (name != 'alpha') else 2
    ax_names = ["x", "y", "z"]
    multipliers = [1, 1, 1]
    if name in ['alpha', 'beta']:
        multipliers = [1100, 850, 1]

    b = b.reshape(3,)
    for i in range(n_axes):
        mult = multipliers[i]*scale
        x_crds = [b[0], b[0] + Base[0, i]*mult]
        y_crds = [b[1], b[1] + Base[1, i]*mult]
        z_crds = [b[2], b[2] + Base[2, i]*mult]
        ax.plot(x_crds, y_crds, z_crds, c=color, marker="^")
        # ax.text(x_crds[1], y_crds[1], z_crds[1], f"{name}{i+1}", c=color, fontsize=10)
        ax.text(x_crds[1], y_crds[1], z_crds[1], f"$\\{name}_{ax_names[i]}$", c=color, fontsize=10)
    return


def fix_axes(ax, n=2.0, c=np.zeros(3)):
    ax.set_xlim3d(c[0] - n, c[0] + n)
    ax.set_ylim3d(c[1] - n, c[1] + n)
    ax.set_zlim3d(c[2] - n, c[2] + n)
    return


if __name__ == "__main__":
    ux_all = sio.loadmat("data/daliborka_01-ux.mat")
    u_all, x_all = ux_all['u'], ux_all['x']
    Q = np.load("data/Q.npy")
    Q_all = np.load("data/Q_all.npy")

    K, R, C = Q2KRC(Q)
    f = K[0, 0] * b1_len
    # f = K[0, 0] * lambd

    # print(f"K = \n{K}")
    # print(f"R = \n{R}")
    # print(f"C = \n{C}")
    # print("-------------------")
    Pb = (1/f) * np.hstack((K@R, -K@R@C))
    print(f"Pb = \n{Pb}")

    R_inv = R.T
    K_inv = np.linalg.inv(K)
    O = np.zeros((3, 1))
    b3_beta = np.array([0, 0, 1]).reshape(3, 1)
    b3_delta = f*R_inv@K_inv@b3_beta
    o = C + b3_delta

    Delta,   d = np.eye(3),         O
    Kappa_old,   k_old = Delta * f,   O
    Epsilon, e = Delta @ R_inv,     C
    Gamma,   g = Epsilon * f,       C
    Kappa,   k = Gamma @ R,         O
    Nu,      n = Epsilon @ K_inv,   C
    Beta,    b = Gamma @ K_inv,     C
    Alpha,   a = Beta[:, 0:2],      o

    sio.savemat('03_bases.mat', {'Pb': Pb, 'f': f,
                                 'Alpha': Alpha, 'a': a,
                                 'Beta': Beta, 'b': b,
                                 'Gamma': Gamma, 'g': g,
                                 'Delta': Delta, 'd': d,
                                 'Epsilon': Epsilon, 'e': e,
                                 'Kappa': Kappa, 'k': k,
                                 'Nu': Nu, 'n': n})

    params = {"Alpha"   : [Alpha,   a, 'g',     "alpha"],
              "Beta"    : [Beta,    b, 'r',     "beta"],
              "Gamma"   : [Gamma,   g, 'b',     "gamma"],
              "Delta"   : [Delta,   d, 'k',     "delta"],
              "Epsilon" : [Epsilon, e, 'm',     "epsilon"],
              "Kappa"   : [Kappa,   k, "brown", "kappa"],
              "Nu"      : [Nu,      n, 'c',     "nu"]}

    bases1 = ["Delta", "Epsilon", "Kappa", "Nu", "Beta"]
    bases2 = ["Alpha", "Beta", "Gamma"]
    # bases2 = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    bases3 = ["Delta", "Epsilon"]

    fig = plt.figure(figsize = (8,8))
    ax = plt.axes(projection='3d')
    fix_axes(ax, 1)
    for i in range(len(bases1)-1):
        ps = params[bases1[i]]
        plot_csystem(ax, ps[0], ps[1], ps[2], ps[3])
    ps = params["Beta"]
    plot_csystem(ax, ps[0], ps[1], ps[2], ps[3], 50)

    ax.scatter(x_all[0, :], x_all[1, :], x_all[2, :], marker='o', s=20, c='b')
    ax.set_xlabel('x [m]', labelpad=20)
    ax.set_ylabel('y [m]', labelpad=20)
    ax.set_zlabel('z [m]', labelpad=20)
    ax.set_title('Coordinate systems $\\delta$, $\\epsilon$, $\\kappa$, $\\nu$ and $\\beta$')
    plt.show()
    # fig.savefig('03_figure1.pdf')


    fig = plt.figure(figsize = (8,8))
    ax = plt.axes(projection='3d')
    fix_axes(ax, 0.01, C)
    for i in range(len(bases2)):
        ps = params[bases2[i]]
        plot_csystem(ax, ps[0], ps[1], ps[2], ps[3])

    # Beta = R_inv*f@K_inv
    u_all_delta = (Beta @ e2p(u_all)) + C
    ax.scatter(u_all_delta[0, :], u_all_delta[1, :], u_all_delta[2, :], marker='o', s=20, c='b')
    # ax.scatter(x_all[0, :], x_all[1, :], x_all[2, :], marker='o', s=20, c='r')
    ax.set_xlabel('x [m]', labelpad=20)
    ax.set_ylabel('y [m]', labelpad=20)
    ax.set_zlabel('z [m]', labelpad=20)
    ax.set_title('Coordinate systems $\\alpha$, $\\beta$ and $\\gamma$')
    plt.show()
    # fig.savefig('03_figure2.pdf')


    fig = plt.figure(figsize = (8, 8))
    ax = plt.axes(projection='3d')
    fix_axes(ax, 1)
    # plot the centres of all Qs
    for Q_new in Q_all:
        _, _, C_new = Q2KRC(Q_new)
        ax.scatter(C_new[0, :], C_new[1, :], C_new[2, :], marker='o', s=4, c='r', alpha=0.25)

    for i in range(len(bases3)):
        ps = params[bases3[i]]
        plot_csystem(ax, ps[0], ps[1], ps[2], ps[3])
    ax.scatter(x_all[0, :], x_all[1, :], x_all[2, :], marker='o', s=20, c='b', alpha=0.5)

    ax.set_xlabel('x [m]', labelpad=20)
    ax.set_ylabel('y [m]', labelpad=20)
    ax.set_zlabel('z [m]', labelpad=20)
    ax.set_title('Estimated camera centres')
    plt.show()
    fig.savefig('03_figure3.pdf')
