import numpy as np                # for matrix computation and linear algebra
import matplotlib.pyplot as plt   # for drawing and image I/O
from mpl_toolkits import mplot3d  # for 3d plots
import matplotlib.image as mpimg
import scipy
import scipy.linalg
import scipy.io as sio            # for matlab file format output
import itertools                  # for generating all combinations
from PIL import Image


def get_line_boundaries(plot_line, img):
    plot_line = plot_line.reshape(1, 3)
    min_x = 1
    min_y = 1
    max_x = img.shape[1]-1
    max_y = img.shape[0]-1

    boundaries = np.array([[min_x, min_x, max_x, max_x],
                           [min_y, max_y, max_y, min_y]])
    boundaries_hom = e2p(boundaries)

    # get line vectors of the boundaries
    a_line = np.cross(boundaries_hom[:, 0], boundaries_hom[:, 1])
    b_line = np.cross(boundaries_hom[:, 1], boundaries_hom[:, 2])
    c_line = np.cross(boundaries_hom[:, 2], boundaries_hom[:, 3])
    d_line = np.cross(boundaries_hom[:, 3], boundaries_hom[:, 0])
    bnd_lines = [a_line, b_line, c_line, d_line]

    line_boundaries = np.zeros([2, 2])
    count = 0
    for bnd_line in bnd_lines:
        line_end = p2e((np.cross(plot_line, bnd_line).reshape(3, 1)))
        x = line_end[0]
        y = line_end[1]
        # plt.plot(x, y, "oy")
        if 1 <= int(x) <= max_x and 1 <= int(y) <= max_y and count < 2:
            line_end = np.reshape(line_end, (1, 2))
            line_boundaries[:, count] = line_end
            count += 1
    return line_boundaries


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


if __name__ == "__main__":
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
    plt.figure()
    plt.plot(np.arange(n_points), errs[0], "b-", label="image 1")
    plt.plot(np.arange(n_points), errs[1], "g-", label="image 2")
    plt.title("Epipolar error for all points")
    plt.xlabel("point index")
    plt.ylabel("epipolar error [px]")
    plt.legend()
    # plt.legend(loc="upper right")
    plt.savefig("08_errors.pdf")
    plt.show()
    plt.close()

    # plot the epipolar lines
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img1)
    ax2.imshow(img2)
    fig.suptitle("The epipolar lines", y=0.85, fontsize=14, fontweight="bold")

    colors = ["darkred", "chocolate", "red", "gold", "darkolivegreen", "lime",
              "steelblue", "royalblue", "navy", "indigo", "orchid", "crimson"]
    for i in range(12):
        u1_sel = u1[:, ix[i]].reshape(2, 1)
        u2_sel = u2[:, ix[i]].reshape(2, 1)

        l1 = F.T @ e2p(u2_sel)
        l2 = F @ e2p(u1_sel)
        l1_b = get_line_boundaries(l1, img1)
        l2_b = get_line_boundaries(l2, img2)

        ax1.scatter(u1_sel[0], u1_sel[1], marker='o', s=10.0, c=colors[i])
        ax1.plot(l1_b[0], l1_b[1], color=colors[i])
        ax1.set_title("Image 1")

        ax2.scatter(u2_sel[0], u2_sel[1], marker='o', s=10.0, c=colors[i])
        ax2.plot(l2_b[0], l2_b[1], color=colors[i])
        ax2.set_title("Image 2")

    plt.savefig("08_eg.pdf")
    plt.show()

    sio.savemat('09a_data.mat', {
        'Fe': ...,
        'E': ...,
        'R': ...,
        'C': ...,
        'P1': ...,
        'P2': ...,
        'X': ...,
        'u1': ...,
        'u2': ...,
        'point_sel_e': points_sel,
    })
