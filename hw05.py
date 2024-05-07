import numpy as np                # for matrix computation and linear algebra
import matplotlib.pyplot as plt   # for drawing and image I/O
import scipy
from scipy import linalg
from mpl_toolkits import mplot3d
import matplotlib.image as mpimg
# from PIL import Image
import scipy.io as sio            # for matlab file format output
import itertools                  # for generating all combinations

TF_COLORS = True

# These coords are in [x, y] format, FLIP IT WHEN INDEXING!!

# points in img1
U0 = np.array([[ 142.4,    93.4,   139.1,   646.9,  1651.4,  1755.2,  1747.3,  1739.5,  1329.2,   972.0],
               [1589.3,   866.7,   259.3,   305.6,    87.3,   624.8,  1093.5,  1593.8,  1610.2,  1579.3]])
# points in img2
U = np.array([[783.8,   462.6,   243.7,   363.9,   465.2,   638.3,   784.7,   954.6,   927.8,   881.4],
              [747.5,   671.2,   586.8,   463.9,   248.5,   260.6,   291.5,   326.9,   412.8,   495.0]])
# missing patch bounds in img2
C = np.array([[474.4,   508.2,   739.3,   737.2],
              [501.7,   347.0,   348.7,   506.7]])
# pokemon sheet bounds in img2
C2 = np.array([[473, 131, 798, 986],
               [226, 597, 807, 325]])


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


# -------------- Main functions -----------------------------------

def u2H(u, u0):
    A = np.zeros((8, 9))
    for i in range(4):
        ui, vi = u[:, i]      # img2 coords
        u_i, v_i = u0[:, i]   # img1 coords
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
    H_best = None
    points_sel = None
    for inx in itertools.combinations(range(0, n_crp), 4):
        H = u2H(u[:, inx], u0[:, inx])
        u_proj = p2e(H @ e2p(u))
        e = np.max(np.linalg.norm(u0 - u_proj, axis=0))
        if e < e_best:
            e_best = e
            H_best = H
            points_sel = np.array(inx)

    return H_best, points_sel


# -------------- Point predicates -----------------------------------

def orient_pred(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

def in_rect(p, C):
    th = -1000  # add some overlap
    c1, c2, c3, c4 = np.hsplit(C, 4)
    b1 = orient_pred(c1, c2, p) >= th
    b2 = orient_pred(c2, c3, p) >= th
    b3 = orient_pred(c3, c4, p) >= th
    b4 = orient_pred(c4, c1, p) >= th

    return b1 and b2 and b3 and b4


# -------------- Matching of color intensities using histograms ---------------------------
# Note: this is not suitable for this task!
I_lvls = 256

def compute_hist(img, Cs=None):
    img_hist = np.zeros((I_lvls, 3))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if Cs is None or not in_rect([j, i], Cs[0]): #  and in_quadr([j, i], Cs[1])):
                intensity = img[i, j]
                for k in range(3):
                    img_hist[intensity[k], k] += 1

    return img_hist

def compute_cdf(img, Cs=None):
    img_hist = compute_hist(img, Cs)
    img_cdf = np.cumsum(img_hist, axis=0)
    img_cdf = img_cdf / img_cdf[-1, :]
    return img_cdf

def match_hists(img, img_target, Cs=None):
    # get both cdfs
    cdf_A = compute_cdf(img)
    cdf_B = compute_cdf(img_target, Cs)
    print("got cdfs")

    # create histogram matching lookup table
    matching_lut = np.zeros((I_lvls, 3))
    for i in range(I_lvls):
        for k in range(3):
            j = 0
            while (j < I_lvls) and (cdf_A[i, k] > cdf_B[j, k]):
                j += 1
            matching_lut[i, k] = j
    print("got matching lut")

    # match the histograms
    img_matched = img
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            intensity = img[i, j]
            for k in range(3):
                img_matched[i, j, k] = matching_lut[intensity[k], k]

    print("got matched img")
    return img_matched

# ---------- using color transformation matrix ------------------------------

def get_color_tf(img1, img2, H, limits):
    x_min, x_max, y_min, y_max = limits

    img2_pixels = []
    img1_pixels = []

    H_inv = np.linalg.inv(H)
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            # small_rect = in_rect([j, i], Cs[0])
            # big_rect = in_rect([j, i], Cs[1])
            # if not small_rect and big_rect:
            if x_min <= j < x_max and y_min <= i < y_max:
                px2 = np.array([j, i]).reshape(2, 1)            # u,x == j, v,y == i
                px1 = np.round(p2e(H @ e2p(px2))).astype(int)
                if 0 <= px1[1] < img1.shape[0] and 0 <= px1[0] < img1.shape[1]:
                    img1_pixels.append(img1[px1[1, 0], px1[0, 0], :])
                    img2_pixels.append(img2[px2[1, 0], px2[0, 0], :])

    img1_pixels = np.array(img1_pixels)
    img2_pixels = np.array(img2_pixels)

    r1, g1, b1 = np.hsplit(img1_pixels, 3)
    r2, g2, b2 = np.hsplit(img2_pixels, 3)

    A = np.c_[r1, g1, b1, np.ones(r1.shape)]
    B = np.c_[r2, g2, b2]
    T = np.linalg.lstsq(A, B, rcond=None)[0]

    return T


if __name__ == "__main__":
    img1 = mpimg.imread("data/pokemon_00.jpg")
    img2 = mpimg.imread("data/pokemon_10.jpg")

    H, points_sel = u2h_optim(U, U0)
    print(points_sel)
    # points_sel = points_sel + 1  # to matlab indexing
    sio.savemat('05_homography.mat', {'u': U, 'u0': U0, "point_sel": points_sel, "H": H})

    if TF_COLORS:
        # __color tf. matrix method__ <-- [better results]
        # limits = x_min, x_max, y_min, y_max
        limits = [550, 800, 550, 700]
        T = get_color_tf(img1, img2, H, limits)
        print(T)

        # Cs = [C, C2]
        # T = get_color_tf(img1, img2, H, Cs)

        # __matching histograms method__ <-- [bad results]
        # try:
        #     img3 = np.load("img3.npy")
        # except:
        #     print("img3.npy doesn't exist --> creating img3")
        #     img3 = match_hists(img1.copy(), img2, Cs)
        #     np.save("img3.npy", img3)

    try:
        if TF_COLORS:
            img2 = mpimg.imread("05_correctedn.png")
        else:
            img2 = mpimg.imread("05_corrected.png")
        print("Loaded")
    except:
        if TF_COLORS:
            print("05_correctedn.png doesn't exist --> creating")
        else:
            print("05_corrected.png doesn't exist --> creating")
        H_inv = np.linalg.inv(H)
        for i in range(img2.shape[0]):
            for j in range(img2.shape[1]):
                if in_rect([j, i], C):
                    px2 = np.array([j, i]).reshape(2, 1)  # u,x == j, v,y == i
                    px1 = np.round(p2e(H @ e2p(px2))).astype(int)
                    if 0 <= px1[1] < img1.shape[0] and 0 <= px1[0] < img1.shape[1]:
                        r, g, b = img1[px1[1, 0], px1[0, 0], :]
                        if TF_COLORS:
                            img2[i, j, :] = (np.array([r, g, b, 1]) @ T).flatten()
                        else:
                            img2[i, j, :] = np.array([r, g, b])

        if TF_COLORS:
            mpimg.imsave("05_correctedn.png", img2)
        else:
            mpimg.imsave("05_corrected.png", img2)

    points_notsel = np.where(~np.isin(np.arange(10), points_sel))
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    ax[0].imshow(img2)
    ax[1].imshow(img1)
    labeled = [False, False]
    for i in range(U.shape[1]):
        if i in points_sel:
            b_box = dict(boxstyle="square", fc="g")
            if labeled[0]:
                ax[0].plot(U[0, i], U[1, i], "go", mec="k", mew=0.5)
            else:
                labeled[0] = True
                ax[0].plot(U[0, i], U[1, i], "go", mec="k", mew=0.5, label="used for H")
            ax[0].annotate(i+1, U[:, i], xytext=(0, 12), textcoords="offset points", color="k", fontweight="bold",
                           bbox=b_box)
            ax[1].plot(U0[0, i], U0[1, i], "go", mec="k", mew=0.5)
            ax[1].annotate(i+1, U0[:, i], xytext=(0, 12), textcoords="offset points", color="k", fontweight="bold",
                           bbox=b_box)
        else:
            b_box = dict(boxstyle="square", fc="r")
            if labeled[1]:
                ax[0].plot(U[0, i], U[1, i], "ro", mec="k", mew=0.5)
            else:
                labeled[1] = True
                ax[0].plot(U[0, i], U[1, i], "ro", mec="k", mew=0.5, label="other points")

            ax[0].annotate(i+1, U[:, i], xytext=(0, 12), textcoords="offset points", color="k", fontweight="bold",
                           bbox=b_box)
            ax[1].plot(U0[0, i], U0[1, i], "ro", mec="k", mew=0.5)
            ax[1].annotate(i+1, U0[:, i], xytext=(0, 12), textcoords="offset points", color="k", fontweight="bold",
                           bbox=b_box)

    ax[0].set_title("Labelled points in my image")
    ax[0].set_xlabel("x [px]")
    ax[0].set_ylabel("y [px]")
    ax[0].legend(loc="lower left")

    ax[1].set_title("Labelled points in reference image")
    ax[1].set_xlabel("x [px]")
    ax[1].set_ylabel("y [px]")

    plt.savefig('05_homography.pdf')
    plt.show()


