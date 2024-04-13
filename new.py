import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy.io as sio


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
    """
    Create function u2H computing homography from four image matches.
    Let u be the image coordinates of points in the first image (2x4 matrix, np.array)
    and u0 (2x4) be the image coordinates of the corresponding points in the second image.
    Then H is a 3x3 homography matrix (np.array), such that
    """
    U1, V1 = np.asarray(u)
    U2, V2 = np.asarray(u0)

    ones, zeros = np.ones_like(U1), np.zeros_like(U1)

    A = np.r_[
        np.c_[U1, V1, ones, zeros, zeros, zeros, -U2 * U1, -U2 * V1, -U2],
        np.c_[zeros, zeros, zeros, U1, V1, ones, -V2 * U1, -V2 * V1, -V2]
    ]

    U, S, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)

    if H[2, 2] != 0:
        H = H / H[2, 2]
    else:
        print('H[2, 2] == 0')

    return H


def dist(H, u, u0):
    """ Function, calculates one-way reprojection error

    Parameters
    ----------
    H : np.array (3, 3)
        Homography matrix
    u : np.array (2, n)
        Image coordinates of points in the first image
    u0 : np.array (2, n)
        Image coordinates of points in the second image

    Returns
    -------
    np.array (N, 1)
        Per-correspondence Eucledian error
    """
    return np.linalg.norm(u0 - p2e(H @ e2p(u)), axis=0)


def find_color_transformation(image1, image2, H, x_min, x_max, y_min, y_max):
    """ Function, finds the color transformation matrix T

    Parameters
    ----------
    image1 : np.array (H, W, 3)
        First image in RGB format
    image2 : np.array (H, W, 3)
        Second image in RGB format
    H : np.array (3, 3)
        Homography matrix
    x_min : int
        Minimum x coordinate of the rectangle
    x_max : int
        Maximum x coordinate of the rectangle
    y_min : int
        Minimum y coordinate of the rectangle
    y_max : int
        Maximum y coordinate of the rectangle

    Returns
    -------
    np.array (4, 3)
        Color transformation matrix
    """
    xs = np.arange(x_min, x_max)
    ys = np.arange(y_min, y_max)

    patch_ref = np.zeros((len(ys), len(xs), 3), dtype=np.uint8)

    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            x_new, y_new = p2e(H @ [x, y, 1]).round().astype(int)
            patch_ref[j, i] = image2[y_new, x_new, :]

    patch_target = image1[y_min:y_max, x_min:x_max, :]

    r = patch_ref[:, :, 0].flatten()
    g = patch_ref[:, :, 1].flatten()
    b = patch_ref[:, :, 2].flatten()

    r_target = patch_target[:, :, 0].flatten()
    g_target = patch_target[:, :, 1].flatten()
    b_target = patch_target[:, :, 2].flatten()

    ones = np.ones_like(r)

    A = np.c_[r, g, b, ones]
    B = np.c_[r_target, g_target, b_target]
    T = np.linalg.lstsq(A, B, rcond=None)[0]

    return T


def fill_missing_pixels(image1, image2, H, T, C):
    """ Function, fills the missing pixels in the second image

    Parameters
    ----------
    image1 : np.array (H, W, 3)
        First image in RGB format
    image2 : np.array (H, W, 3)
        Second image in RGB format
    H : np.array (3, 3)
        Homography matrix
    T : np.array (4, 3)
        Color transformation matrix
    C : np.array (2, N)
        Coordinates of black rectangle in the first image

    Returns
    -------
    np.array (H, W, 3)
        Second image with filled missing pixels
    """
    image1 = image1.copy()
    x_min, x_max = int(np.min(C[0])), int(np.max(C[0]) + 1)
    y_min, y_max = int(np.min(C[1])), int(np.max(C[1]) + 1)

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            if image1[y, x, :].sum() > 500:
                continue
            x_new, y_new = p2e(H @ [x, y, 1]).round().astype(int)
            r, g, b = image2[y_new, x_new].flatten()
            image1[y, x, :] = [r, g, b, 1] @ T
    return image1


def u2h_optim(u, u0):
    u = np.asarray(u)
    u0 = np.asarray(u0)

    indices = np.arange(u.shape[1])
    min_transfer_error = np.inf

    for idx in itertools.combinations(indices, 4):

        H = u2H(u[:, idx], u0[:, idx])

        errors = dist(H, u, u0)

        transfer_error = errors.max()

        if transfer_error < min_transfer_error:
            min_transfer_error = transfer_error
            print(min_transfer_error)
            H_best = H
            idx_best = idx

    return H_best, idx_best


if __name__ == '__main__':

    # ! 1
    # Implement function u2H. Use the following test data:
    u = [
        [0, 0, 1, 1],
        [0, 1, 1, 0]
    ]

    u0 = [
        [1, 2, 1.5, 1],
        [1, 2, 0.5, 0]
    ]

    H = u2H(u, u0)

    print(H)
    print()

    # ! 2 - 4
    u0 = np.array([[ 142.4,    93.4,   139.1,   646.9,  1651.4,  1755.2,  1747.3,  1739.5,  1329.2,   972.0],
                   [1589.3,   866.7,   259.3,   305.6,    87.3,   624.8,  1093.5,  1593.8,  1610.2,  1579.3]])

    u = np.array([[783.8,   462.6,   243.7,   363.9,   465.2,   638.3,   784.7,   954.6,   927.8,   881.4],
                  [747.5,   671.2,   586.8,   463.9,   248.5,   260.6,   291.5,   326.9,   412.8,   495.0]])

    C = np.array([[474.4,   508.2,   739.3,   737.2],
                  [501.7,   347.0,   348.7,   506.7]])

    H, point_sel = u2h_optim(u, u0)

    image1 = plt.imread('pokemon_10.jpg').copy()
    image2 = plt.imread('pokemon_00.jpg').copy()

    # ! 5
    # Store the matches as u, u0, the array of indices of the four matches used to compute the homography
    # as point_sel and the homography matrix H in 05_homography.mat file.
    sio.savemat('05_homography.mat', {'H': H, 'u': u, 'u0': u0, 'point_sel': point_sel})

    # ! 6
    # Fill the pixels of the black square in your image using the pixels from the reference image mapped by H.
    # The pixels can be found e.g. by an image intensity thresholding,
    # or coordinates C of the square corners from Input Data can be used.
    # Store the corrected bitmap image as 05_corrected.png.
    # Optionally, try some colour normalization of filled-in area (up to one bonus point).

    x_min, x_max = 350, 450
    y_min, y_max = 450, 550

    T = find_color_transformation(image1, image2, H, x_min, x_max, y_min, y_max)
    image3 = fill_missing_pixels(image1, image2, H, T, C)

    from PIL import Image

    Image.fromarray(image3).save('05_corrected.png')

    # ! 7
    # Display both images side by side and draw the image matches to both as crosses with labels (1 to 10)
    # and highlight the four points used for computing the best H. Export as 05_homography.pdf.

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_title('Labelled points in my image')
    ax[1].set_title('Labelled points in reference image')
    ax[0].set_xlabel('x [px]')
    ax[1].set_xlabel('x [px]')
    ax[0].set_ylabel('y [px]')
    ax[1].set_ylabel('y [px]')
    ax[0].imshow(image1)
    ax[1].imshow(image2)

    for i, (x, y) in enumerate(u.T):
        c = 'm' if i in point_sel else 'r'
        marker = 'o' if i in point_sel else 'x'
        ax[0].annotate(i, (x, y), bbox=dict(boxstyle='circle, pad=.2', fc=c, ec='none'),
                       textcoords="offset points", xytext=(0, 12), ha='center', va='center')
        ax[0].scatter(x, y, c=c, marker=marker, s=20)

    for i, (x, y) in enumerate(u0.T):
        c = 'm' if i in point_sel else 'r'
        marker = 'o' if i in point_sel else 'x'
        ax[1].annotate(i, (x, y), bbox=dict(boxstyle='circle, pad=.2', fc=c, ec='none'),
                       textcoords="offset points", xytext=(0, 12), ha='center', va='center')
        ax[1].scatter(x, y, c=c, marker=marker, s=20)
    plt.savefig('05_homography.pdf')
    plt.show()
    plt.close()