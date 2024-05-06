import numpy as np                # for matrix computation and linear algebra
import matplotlib.pyplot as plt   # for drawing and image I/O
from mpl_toolkits import mplot3d  # for 3d plots
import matplotlib.image as mpimg
from PIL import Image
import scipy.io as sio            # for matlab file format output
import itertools                  # for generating all combinations
from SeqWriter import SeqWriter

PLOT_1 = False

C1 = np.array([[474.4,   508.2,   739.3,   737.2],
               [501.7,   347.0,   348.7,   506.7]])

C2 = np.array([[572.6,   385.9,   572.4,   756.5],
               [542.8,   421.6,   334.8,   435.6]])

Co1 = np.array([[120.1,   464.7,   989.8,   799.0],
                [598.9,   219.7,   322.2,   814.5]])

Co2 = np.array([[398.4,   154.8,   702.0,  1105.9],
                [823.7,   346.3,   242.6,   583.8]])


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


def get_lowerright(p1, p2, p3, p4):
    P = np.hstack((p1, p2, p3, p4))
    vals = P[0]*P[1]
    return P[:, np.argmax(vals)]


def plot_line(p1, p2, c="k-"):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c)


def get_angle_cos(v, vo, K):
    K_inv = np.linalg.inv(K)
    omega = K_inv.T@K_inv
    v_norm, vo_norm = np.linalg.norm((K_inv@v, K_inv@vo), axis=1).flatten()  # just two norms, nothing special
    cos_a = (v.T@omega@vo).item() / (v_norm*vo_norm)
    return cos_a


if __name__ == "__main__":
    # mistake: only SOME of the origin corners work, but why, is a mystery
    #          mistake was in unknowingly choosing a wrong origin but looking for a bug elsewhere
    #          also, swapping the x, y axes flips the cube
    titles = ["pokemon_10.jpg", "pokemon_19.jpg"]
    img1 = mpimg.imread("data/pokemon_10.jpg")
    img2 = mpimg.imread("data/pokemon_19.jpg")
    imgs = [img1, img2]

    # --- task 1 ---
    h, w = img1.shape[:2]
    l_in, l_out = 0, 0
    vp = []
    for i, (C, Co) in enumerate(zip([C1, C2], [Co1, Co2])):
        l_in = np.argmax(C[1])   # index of the lowest inner point
        l_out = np.argmax(Co[1]) # index of the lowest outer point
        C, Co = e2p(C), e2p(Co)
        L =  [cross( C[:, i%4], C[:, (i+1)%4]) for i in range(l_in, l_in + 4)]
        Lo = [cross(Co[:, i%4], Co[:, (i+1)%4]) for i in range(l_out, l_out + 4)]
        L, Lo = np.hstack(L), np.hstack(Lo)  # why the deprecation warning though, works ok

        v1 =  norm(cross( L[:, 0],  L[:, 2]))
        v2 =  norm(cross( L[:, 1],  L[:, 3]))
        vo1 = norm(cross(Lo[:, 0], Lo[:, 2]))
        vo2 = norm(cross(Lo[:, 1], Lo[:, 3]))
        V = np.hstack((v1, v2, vo1, vo2))
        v_left, v_right = V[:, np.argmin(V[0])], V[:, np.argmax(V[0])]
        # v_line = cross(v_left, v_right)
        vp.append([v1, v2, vo1, vo2])

        if PLOT_1:
            plt.imshow(imgs[i])

            # plot vanishing points and vanishing line
            plot_line(v1, v2, "bx")
            plot_line(vo1, vo2, "rx")
            plot_line(v_left, v_right, "g-")

            # plot outer lines
            plot_line(Co[:, l_out], vo1, "r-")
            plot_line(Co[:, l_out], vo2, "r-")
            plot_line(Co[:, (l_out - 1) % 4], vo1, "r-")
            plot_line(Co[:, (l_out + 1) % 4], vo2, "r-")

            # plot inner lines
            plot_line(C[:, l_in], v1, "b-")
            plot_line(C[:, l_in], v2, "b-")
            plot_line(C[:, (l_in - 1) % 4], v1, "b-")
            plot_line(C[:, (l_in + 1) % 4], v2, "b-")

            plt.title(titles[i])
            zoom = False
            if zoom:
                plt.xlim([-250, w+250])
                plt.ylim([h+250, -250])
                plt.savefig(f"07_vp{i+1}_zoom.pdf")
            else:
                plt.ylim([2000, -2000])
                plt.xlim([v_left[0], v_right[0]])
                plt.savefig(f"07_vp{i+1}.pdf")
            plt.show()

    vp1, vp2 = vp


    # --- task 2 ---
    # compute the K using equations 11.41
    # projection rays (v1 and v2) of van. pt. pairs are orthogonal, so
    #   v1.T@Ω@v2 = 0
    # Ω has only 3 params o13, o23 and o33, ve use 3 v. p. pairs to compute them

    v_pairs = [vp1[:2], vp1[2:], vp2[:2]]
    A = np.zeros((3, 3))
    b = np.zeros((3, 1))
    for i, [v, vo] in enumerate(v_pairs):
        [v11, v12, v13] = v.flatten()
        [v21, v22, v23] = vo.flatten()
        A[i, :] = [v23*v11 + v21*v13, v23*v12 + v22*v13, v23*v13]
        b[i, :] = -(v21*v11 + v12*v22)

    [o13, o23, o33] = np.linalg.solve(A, b).flatten().tolist()
    k13 = -o13
    k23 = -o23
    k11 = np.sqrt(o33 - k13**2 - k23**2)

    K = np.array([[k11,   0, k13],
                  [  0, k11, k23],
                  [  0,   0,   1]])
    print(K)
    K_inv = np.linalg.inv(K)
    omega = K_inv.T@K_inv

    angles = []

    v_pairs = [[vp1[0], vp1[2]], [vp1[1], vp1[3]], [vp2[0], vp2[2]], [vp2[1], vp2[3]]]
    for (v, vo) in v_pairs:
        cos_a = get_angle_cos(v, vo, K)
        angles.append(np.arccos(cos_a))

    angle = np.mean(np.array(angles))
    print("angle: ", angle/np.pi*180)


    # --- task 3 ---
    from hw04a import p3p_distances, get_cosines
    from hw04b import p3p_RC

    # first image
    l_in = np.argmax(C1[1])   # index of the lowest inner point
    idxs = [(l_in-1)%4, l_in, (l_in+1)%4]
    # idxs = [(l_in-2)%4, (l_in-1)%4, l_in]
    # idxs = [l_in, (l_in+1)%4, (l_in+2)%4]
    # idxs = [(l_in+1)%4, (l_in+2)%4, (l_in+3)%4]
    # idxs = [(l_in+2)%4, (l_in+3)%4, (l_in+4)%4]
    u = e2p(C1[:, idxs])
    x1, x2, x3, = np.hsplit(u, 3)
    c12, c23, c31 = get_cosines(x1, x2, x3, K)
    d12, d23, d31 = 1, 1, np.sqrt(2)  # we def. side of the black sqare as unit
    n1s, n2s, n3s = p3p_distances(d12, d23, d31, c12, c23, c31)
    N = [n1s[0], n2s[0], n3s[0]]

    u = p2e(u)
    X = np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0, 0, 0]])

    R_new, C_new = p3p_RC(N, u, X, K)

    P1 = np.hstack([K@R_new, -K@R_new@C_new])
    X_cube = np.array([[0, 1, 1, 0, 0, 1, 1, 0],
                       [0, 0, 1, 1, 0, 0, 1, 1],
                       [0, 0, 0, 0, 1, 1, 1, 1]])
    u_cube = p2e(P1 @ e2p(X_cube))
    plt.imshow(img1)
    for i in range(X_cube.shape[1]):
        idxs = [(i-1)%4 + (i//4)*4, (i+1)%4 + (i//4)*4, (i+4)%8]  # just neighbors to i in X_cube
        for idx in idxs:
            plt.plot(u_cube[0, [i, idx]], u_cube[1, [i, idx]], "b-")
    plt.plot(u_cube[0, 0], u_cube[1, 0], "bo")  # origin
    # plt.plot(u_cube[0, 1], u_cube[1, 1], "ro")  # x-dir
    # plt.plot(u_cube[0, 3], u_cube[1, 3], "bo")  # y-dir
    # plt.plot(u_cube[0, 4], u_cube[1, 4], "go")  # z-dir
    plt.savefig("07_box_wire1.pdf")
    plt.show()


    # second image
    l_in = np.argmax(C2[1])   # index of the lowest inner point
    idxs = [(l_in-2)%4, (l_in-1)%4, l_in]
    u = e2p(C2[:, idxs])
    x1, x2, x3, = np.hsplit(u, 3)
    c12, c23, c31 = get_cosines(x1, x2, x3, K)
    d12, d23, d31 = 1, 1, np.sqrt(2)  # we def. side of the black sqare as unit
    n1s, n2s, n3s = p3p_distances(d12, d23, d31, c12, c23, c31)
    N = [n1s[0], n2s[0], n3s[0]]

    u = p2e(u)
    X = np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0, 0, 0]])

    R2_new, C2_new = p3p_RC(N, u, X, K)

    P2 = np.hstack([K@R2_new, -K@R2_new@C2_new])
    X_cube = np.array([[0, 1, 1, 0, 0, 1, 1, 0],
                       [0, 0, 1, 1, 0, 0, 1, 1],
                       [0, 0, 0, 0, 1, 1, 1, 1]])
    u_cube = p2e(P2 @ e2p(X_cube))
    plt.imshow(img2)
    for i in range(X_cube.shape[1]):
        idxs = [(i-1)%4 + (i//4)*4, (i+1)%4 + (i//4)*4, (i+4)%8]  # just neighbors to i in X_cube
        for idx in idxs:
            plt.plot(u_cube[0, [i, idx]], u_cube[1, [i, idx]], "b-")
    plt.plot(u_cube[0, 0], u_cube[1, 0], "bo")  # origin
    # plt.plot(u_cube[0, 1], u_cube[1, 1], "ro")  # x-dir
    # plt.plot(u_cube[0, 3], u_cube[1, 3], "bo")  # y-dir
    # plt.plot(u_cube[0, 4], u_cube[1, 4], "go")  # z-dir
    plt.savefig("07_box_wire2.pdf")
    plt.show()

    vp1 = p2e(np.hstack(vp1))
    vp2 = p2e(np.hstack(vp2))

    sio.savemat('07_data.mat', {
        'u1': np.hstack([C1, Co1]),
        'u2': np.hstack([C2, Co2]),
        'vp1': vp1,
        'vp2': vp2,
        'K': K,
        'angle': angle,
        'C1': C_new,
        'C2': C2_new,
        'R1': R_new,
        'R2': R2_new,
    })

    import cv2, os, scipy, shutil

    writer = SeqWriter('07_seq_wire.avi')
    h, w = img1.shape[:2]

    # Generate a sequence of 20 virtual views on the cube, interpolating camera from the first image to the second.
    # Use the first image, transformed by a homography.
    # Store the middle image of the sequence as 07_box_wire3.pdf and whole sequence as 07_seq_wire.avi.

    os.makedirs('cache', exist_ok=True)

    n_steps = 20
    for step, lambd in enumerate(np.linspace(0, 1, n_steps)):
        Ci = C2_new * lambd + C_new * (1-lambd)  # C1 --> C2
        Ri = scipy.linalg.fractional_matrix_power(R2_new @ R_new.T, lambd).real @ R_new  # R1 --> R2
        Pi = np.hstack([K@Ri, -K@Ri@Ci])

        # interpolated homography (8.21 and 8.22)
        G = P1[:, [0, 1, 3]]
        G_prime = Pi[:, [0, 1, 3]]
        G_prime_inv = np.linalg.inv(G_prime)
        H = G @ G_prime_inv  # Eq. (8.26)

        grid = np.meshgrid(range(w), range(h))
        img_coords = np.concatenate(grid).reshape(2, -1)

        # map coordinates by the homography
        xs, ys = p2e(H @ e2p(img_coords)).round().astype(int)

        # filter out the out-of-bound coordinates
        mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs, ys = xs[mask], ys[mask]
        i, j = img_coords[:2, mask]

        # map image pixels
        img3 = np.zeros_like(img1)
        img3[j, i] = img1[ys, xs]

        # plot the cube again
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img3)
        X_cube = np.array([[0, 1, 1, 0, 0, 1, 1, 0],
                           [0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1]])

        u_cube = p2e(Pi @ e2p(X_cube))
        plt.imshow(img3)
        for i in range(X_cube.shape[1]):
            idxs = [(i-1)%4 + (i//4)*4, (i+1)%4 + (i//4)*4, (i+4)%8]  # just neighbors to i in X_cube
            for idx in idxs:
                plt.plot(u_cube[0, [i, idx]], u_cube[1, [i, idx]], "b-")

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'cache/{step}.png', bbox_inches='tight', pad_inches=0)
        if step == (n_steps // 2):
            plt.savefig('07_box_wire3.pdf', bbox_inches='tight', pad_inches=0)
        im = cv2.imread(f'cache/{step}.png')
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        writer.Write(im)
        # plt.show()
        print(f"Got image {step+1}/{n_steps}")
        plt.close()

    writer.Close()

    # get rid of the temporary directory
    shutil.rmtree('cache')
