import numpy as np                # for matrix computation and linear algebra
import matplotlib.pyplot as plt   # for drawing and image I/O
import scipy
from scipy import linalg
from mpl_toolkits import mplot3d
import matplotlib.image as mpimg
from PIL import Image
import scipy.io as sio            # for matlab file format output
import itertools                  # for generating all combinations

from hw05 import u2h_optim


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


if __name__ == "__main__":
    # mistakes: took the width, height values from cw and didn't check
    #           though u2h_optim produces H_ij, but produced H_ji
    imgs = []
    for i in range(1, 8):
        imgs.append(mpimg.imread(f"data/bridge_0{i}.jpg"))
    # correspondences of pairs {i, i+1}
    corresp = sio.loadmat("data/bridge_corresp.mat")['u']

    img_h, img_w = imgs[0].shape[:2]
    corners = np.array([[0, img_w, img_w,     0, 0],
                        [0,     0, img_h, img_h, 0]])

    # --- part 1 ---
    plt.figure()
    Hs = np.empty(7, dtype=object)
    errors = np.empty(7, dtype=object)
    for i in range(0, 6):
        u  = corresp[i, i+1]  # source image points
        u0 = corresp[i+1, i]  # target image points
        Hs[i] = u2h_optim(u, u0)[0]
        errors[i] = np.sort(np.linalg.norm(p2e(Hs[i] @ e2p(u)) - u0, axis=0))
        plt.plot(np.arange(1, 11), errors[i], label=f"{i+1}--{i+2}")

    plt.xlim(0, 10)
    plt.ylim(0, 8)

    plt.title('Transfer errors')
    plt.ylabel('err [px]')
    plt.xlabel('point rank (min. err)')
    plt.legend(loc="upper left")
    plt.savefig('06_errors.pdf')
    plt.show()

    # --- part 2 ---
    H_i4 = np.array(7 * [np.eye(3)])
    H_4i = np.array(7 * [np.eye(3)])
    for i in range(3):
        # H_14 = inv(H_43) @ int(H_32) @ inv(H_21) @ I  ->  j = {0, 1, 2}
        for j in range(i, 3):
            H_i4[i] = np.linalg.inv(Hs[j]) @ H_i4[i]
        H_4i[i] = np.linalg.inv(H_i4[i])

        # H_74 = H_54 @ H_65 @ H_76 @ I  ->  j = {5, 4, 3}
        for j in range(5-i, 2, -1):
            H_i4[6-i] = Hs[j] @ H_i4[6-i]
        H_4i[6-i] = np.linalg.inv(H_i4[6-i])


    plt.figure()
    txt_params = {'fontsize'            : 10,
                  'color'               : "black",
                  'verticalalignment'   : "center",
                  'horizontalalignment' : "center",
                  'fontweight'          : "heavy",
                  'bbox'                : dict(facecolor='red', alpha=0.3)}
    for i in range(1, 6):
        corners_i = p2e(H_4i[i] @ e2p(corners))
        plt.plot(corners_i[0, :], corners_i[1, :])
        plt.text(np.mean(corners_i[0, :-1]), np.mean(corners_i[1, :-1]), i+1, **txt_params)

    # plt.xlim(0, 10)
    # plt.ylim(0, 8)

    plt.title('Image borders in image 4 plane')
    plt.ylabel('y [px]')
    plt.xlabel('x [px]')
    plt.gca().invert_yaxis()
    plt.savefig('06_borders.pdf')
    plt.show()

    # find the global borders for the panorama
    pn_x_min, pn_x_max = np.inf, -np.inf
    pn_y_min, pn_y_max = np.inf, -np.inf
    for i in range(2, 5):
        corners_i = p2e(H_4i[i] @ e2p(corners))
        pn_x_min = np.min((pn_x_min, np.min(corners_i[0, :])))
        pn_x_max = np.max((pn_x_max, np.max(corners_i[0, :])))
        pn_y_min = np.min((pn_y_min, np.min(corners_i[1, :])))
        pn_y_max = np.max((pn_y_max, np.max(corners_i[1, :])))
    pn_x_min, pn_x_max = int(np.floor(pn_x_min)), int(np.ceil(pn_x_max))
    pn_y_min, pn_y_max = int(np.floor(pn_y_min)), int(np.ceil(pn_y_max))
    pn_w = int(pn_x_max - pn_x_min)
    pn_h = int(pn_y_max - pn_y_min)

    xs, ys = np.meshgrid(np.arange(pn_x_min, pn_x_max), np.arange(pn_y_min, pn_y_max))
    pn_pts = np.vstack((xs.flatten(), ys.flatten())).astype(int)
    pn_idxs = pn_pts - np.array([pn_x_min, pn_y_min]).reshape(2, 1)

    pn_img = np.zeros((pn_h, pn_w, 3), dtype=np.uint8)
    for i in range(2, 5):
        pn_pts_i = np.round(p2e(H_i4[i] @ e2p(pn_pts))).astype(int)
        in_mask = (0 <= pn_pts_i[0, :]) * (pn_pts_i[0, :] < img_w) * \
                  (0 <= pn_pts_i[1, :]) * (pn_pts_i[1, :] < img_h)
        pn_pts_i = pn_pts_i[:, in_mask]
        pn_idxs_i = pn_idxs[:, in_mask]
        img_i = imgs[i]
        pn_img[pn_idxs_i[1], pn_idxs_i[0]] = img_i[pn_pts_i[1], pn_pts_i[0]]

    plt.imshow(pn_img)
    plt.show()
    # Image.fromarray(pn_img).save('06_panorama.png')

    # --- part 3 ---
    exif_w = 2400
    exif_h = 1800
    x_res = 2160000 / 225
    y_res = 1611200 / 168
    f = 7.4  # [mm]
    inch2mm = 25.4  # [mm]

    fx = f * x_res / inch2mm
    fy = f * y_res / inch2mm
    x0 = exif_w / 2
    y0 = exif_h / 2

    scale = 0.5
    K = np.array([[fx, 0, x0],
                  [0, fy, y0],
                  [0,  0,  1]])
    K = K * scale
    K[2, 2] = 1

    sio.savemat('06_data.mat', {'K': K})

    density = 20
    b1 = np.vstack((np.linspace(0, img_w, density),  np.zeros(density)))
    b2 = np.vstack((np.ones(density)*img_w,  np.linspace(0, img_h, density)))
    b3 = np.vstack((np.linspace(img_w, 0, density),  np.ones(density)*img_h))
    b4 = np.vstack((np.zeros(density),  np.linspace(img_h, 0, density)))
    borders = np.hstack((b1, b2, b3, b4, b1))

    plt.figure()
    txt_params = {'fontsize'            : 10,
                  'color'               : "black",
                  'verticalalignment'   : "center",
                  'horizontalalignment' : "center",
                  'fontweight'          : "heavy",
                  'bbox'                : dict(facecolor='red', alpha=0.3)}
    K_inv = np.linalg.inv(K)
    borders_cyl_all = []
    for i in range(7):
        borders_i = K_inv @ H_4i[i] @ e2p(borders)
        xs, ys, zs = borders_i
        a = np.arctan2(xs, zs)
        y = ys / np.sqrt(xs**2 + zs**2)
        borders_cyl_i = K[0, 0]*np.array([a, y])
        plt.plot(borders_cyl_i[0], borders_cyl_i[1])
        plt.text(np.mean(borders_cyl_i[0, :-1]), np.mean(borders_cyl_i[1, :-1]), i+1, **txt_params)

        borders_cyl_all.append(borders_cyl_i)

    # plt.xlim(0, 10)
    plt.ylim(-1500, 1500)

    plt.title('Image borders in the cylindrical plane')
    plt.ylabel('y [px]')
    plt.xlabel('x [px]')
    plt.gca().invert_yaxis()
    plt.savefig('06_borders_c.pdf')
    plt.show()

    # find the global borders for the panorama again
    pn_x_min, pn_x_max = np.inf, -np.inf
    pn_y_min, pn_y_max = np.inf, -np.inf
    for i in range(7):
        borders_cyl_i = borders_cyl_all[i]
        pn_x_min = np.min((pn_x_min, np.min(borders_cyl_i[0, :])))
        pn_x_max = np.max((pn_x_max, np.max(borders_cyl_i[0, :])))
        pn_y_min = np.min((pn_y_min, np.min(borders_cyl_i[1, :])))
        pn_y_max = np.max((pn_y_max, np.max(borders_cyl_i[1, :])))
    pn_x_min, pn_x_max = int(np.floor(pn_x_min)), int(np.ceil(pn_x_max))
    pn_y_min, pn_y_max = int(np.floor(pn_y_min)), int(np.ceil(pn_y_max))
    pn_w = int(pn_x_max - pn_x_min)
    pn_h = int(pn_y_max - pn_y_min)

    xs, ys = np.meshgrid(np.arange(pn_x_min, pn_x_max), np.arange(pn_y_min, pn_y_max))
    pn_pts = np.vstack((xs.flatten(), ys.flatten())).astype(int)
    pn_idxs = pn_pts - np.array([pn_x_min, pn_y_min]).reshape(2, 1)

    pn_img = np.zeros((pn_h, pn_w, 3), dtype=np.uint8)

    xs, ys = np.meshgrid(np.arange(img_w), np.arange(img_h))
    img_idxs = np.vstack((xs.flatten(), ys.flatten())).astype(int)

    for i in range(7):
        img_idxs_i = K_inv @ H_4i[i] @ e2p(img_idxs)
        xs, ys, zs = img_idxs_i

        a = np.arctan2(xs, zs)
        y = ys / np.sqrt(xs**2 + zs**2)

        pn_idxs_cyl = K[0, 0]*np.array([a, y]) - np.hstack([pn_x_min, pn_y_min]).reshape(2, 1)
        pn_idxs_cyl = np.round(pn_idxs_cyl).astype(int)

        img_i = imgs[i]
        pn_img[pn_idxs_cyl[1], pn_idxs_cyl[0]] = img_i[img_idxs[1], img_idxs[0]]

    plt.imshow(pn_img)
    plt.show()
    # Image.fromarray(pn_img).save('06_panorama_c.png')
