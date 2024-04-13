import numpy as np                # for matrix computation and linear algebra
import matplotlib.pyplot as plt   # for drawing and image I/O
import scipy
from scipy import linalg
from mpl_toolkits import mplot3d
import matplotlib.image as mpimg
# from PIL import Image
import scipy.io as sio            # for matlab file format output
import itertools                  # for generating all combinations


U0 = np.array([[ 142.4,    93.4,   139.1,   646.9,  1651.4,  1755.2,  1747.3,  1739.5,  1329.2,   972.0],
               [1589.3,   866.7,   259.3,   305.6,    87.3,   624.8,  1093.5,  1593.8,  1610.2,  1579.3]])

U = np.array([[783.8,   462.6,   243.7,   363.9,   465.2,   638.3,   784.7,   954.6,   927.8,   881.4],
              [747.5,   671.2,   586.8,   463.9,   248.5,   260.6,   291.5,   326.9,   412.8,   495.0]])

C = np.array([[474.4,   508.2,   739.3,   737.2],
              [501.7,   347.0,   348.7,   506.7]])


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


def u2H(u, u0):
    A = np.zeros((8, 9))
    for i in range(4):
        ui, vi = u[:, i]
        u_i, v_i = u0[:, i]
        A[2*i, :]   = np.array([ui, vi, 1,  0,  0, 0, -u_i*ui, -u_i*vi, -u_i])
        A[2*i+1, :] = np.array([ 0,  0, 0, ui, vi, 1, -v_i*ui, -v_i*vi, -v_i])

    [U, S, VT] = np.linalg.svd(A)
    H = VT[-1, :].reshape(3, 3)
    assert H[2, 2] != 0, "H[2, 2] is 0!"
    H /= H[2, 2]   # H[2, 2] should be 1
    return H


def u2h_optim(u, u0):
    n_crp = u.shape[1]
    e_best = np.inf
    H_best = np.zeros((3, 3))
    points_sel = np.array([])
    for inx in itertools.combinations(range(0, n_crp), 4):
        H = u2H(u[:, inx], u0[:, inx])
        u_proj = p2e(H @ e2p(u))
        e = np.max(np.linalg.norm(u0 - u_proj, axis=0))
        if e < e_best:
            print(e)
            e_best = e
            H_best = H
            points_sel = np.array(inx)
    return H_best, points_sel


if __name__ == "__main__":
    img1 = mpimg.imread("pokemon_00.jpg")
    img2 = mpimg.imread("pokemon_10.jpg")
    H, points_sel = u2h_optim(U, U0)
    points_sel = points_sel + 1  # to matlab indexing
    sio.savemat('05_homography.mat', {'u': U, 'u0': U0, "point_sel": points_sel, "H": H})

    H_inv = np.linalg.inv(H)
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if np.sum(img2[i, j, :]) <= 40:
                px2 = np.array([i, j]).reshape(2, 1)
                px1 = np.round(p2e(H @ e2p(px2))).astype(int)
                # print(px1[0], px1[1])
                if 0 <= px1[0] < img1.shape[0] and 0 <= px1[1] < img1.shape[1]:
                    img2[i, j, :] = img1[px1[0], px1[1], :]
                    # img1[px1[0], px1[1], :] = [0, 0, 0]

    plt.figure()
    plt.imshow(img2)
    U0_proj = p2e(H_inv @ e2p(U0))
    plt.plot(C[0], C[1], "rx")
    plt.plot(U[0], U[1], "go")
    plt.plot(U0_proj[0], U0_proj[1], "mo")
    plt.show()

    plt.imshow(img1)
    U_proj = p2e(H @ e2p(U))
    plt.plot(U0[0], U0[1], "go")
    plt.plot(U_proj[0], U_proj[1], "mo")
    plt.show()

    mpimg.imsave("05_corrected.png", img2)

