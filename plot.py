import matplotlib.pyplot as plt
import numpy as np


def plot_rp_errors(errs, name="09_errorsr.pdf"):
    n_points = max(errs[0].shape)
    plt.figure()

    plt.plot(np.arange(n_points), errs[0], "b-", label="image 1")
    plt.plot(np.arange(n_points), errs[1], "g-", label="image 2")
    plt.title("Reprojection error for all points")
    plt.xlabel("point index")
    plt.ylabel("reprojection error [px]")
    plt.legend()
    plt.savefig(name)
    plt.show()
    plt.close()


def plot_reprojections(img1, img2, u1, u2, u1_proj, u2_proj, edges):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img1)
    ax2.imshow(img2)
    fig.suptitle("The reprojections", y=0.85, fontsize=14, fontweight="bold")

    # plot the edges on bottom
    for i in range(edges.shape[1]):
        e = edges[:, i]
        ax1.plot([u1[0, e[0]], u1[0, e[1]]], [u1[1, e[0]], u1[1, e[1]]], "y-")
        ax2.plot([u2[0, e[0]], u2[0, e[1]]], [u2[1, e[0]], u2[1, e[1]]], "y-")

    # note: use zorder to put the scatter on top
    ax1.set_title("Image 1")
    ax1.plot(u1[0], u1[1], "bo", markersize=2)
    ax1.scatter(u1_proj[0], u1_proj[1], marker="o", facecolors="none", edgecolors="r", zorder=2)

    ax2.set_title("Image 2")
    ax2.plot(u2[0], u2[1], "bo", markersize=2)
    ax2.scatter(u2_proj[0], u2_proj[1], marker="o", facecolors="none", edgecolors="r", zorder=2)

    plt.savefig("09_reprojection.pdf")
    plt.show()


def plot_ep_errors(errs, name="some_ep_errors.pdf"):
    n_points = max(errs[0].shape)
    plt.figure()

    plt.plot(np.arange(n_points), errs[0], "b-", label="image 1")
    plt.plot(np.arange(n_points), errs[1], "g-", label="image 2")
    plt.title("Epipolar error for all points")
    plt.xlabel("point index")
    plt.ylabel("epipolar error [px]")
    plt.legend()
    # plt.legend(loc="upper right")
    plt.savefig(name)
    plt.show()
    plt.close()


def plot_ep_lines(img1, img2, u1, u2, ix, F, name="some_ep_lines.pdf"):
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

    plt.savefig(name)
    plt.show()


# my function from TDV course
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
