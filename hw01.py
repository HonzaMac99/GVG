import numpy as np                # for matrix computation and linear algebra
import matplotlib.pyplot as plt   # for drawing and image I/O
import matplotlib.image as mpimg
# from PIL import Image
import scipy.io as sio            # for matlab file format output
import itertools                  # for generating all combinations

PRINT = False

u2 = np.array([[ -182.6,  -170.5,  -178.8,  -202.6,    51.5,   -78.0,   106.1],
               [  265.8,   447.0,   486.7,   851.9,   907.1,  1098.7,  1343.6]])

u = np.array([[ 147.0,  275.0,  300.0,  551.8,   643.8,  749.8,  961.2],
              [ 186.0,  213.0,  228.4,  320.4,   153.8,  282.9,  202.1]])

colors = np.array([[255, 0, 0],
                   [0, 255, 0],
                   [0, 0, 255],
                   [255, 0, 255],
                   [0, 255, 255],
                   [255, 255, 0],
                   [255, 255, 255]])

def e2p(u):
    n = u.shape[1]
    return np.vstack((u, np.ones((1, n))))


# compute A from the coresp. triplet
def compute_A(u2, u):
    b = np.vstack((u[:, 0], u[:, 1], u[:, 2])).reshape(6, 1)

    A_u2 = np.zeros((6, 6))
    for i in range(3):
        A_u2[i*2, :]   = np.array([u2[0, i], u2[1, i], 1, 0, 0, 0])
        A_u2[i*2+1, :] = np.array([0, 0, 0, u2[0, i], u2[1, i], 1])

    # solve Ax = b
    x = np.linalg.solve(A_u2, b)
    A = np.vstack((x[:3].T, x[3:].T))
    return A


def estimate_A(u2, u):
    n = u.shape[1]
    iter = itertools.combinations(range(0, n), 3)  # three of n

    # iterate all combinations
    e_max_best = np.inf
    A_best = np.zeros((2, 3))
    for inx in iter:
        u_i = u[:, inx]
        u2_i = u2[:, inx]

        # compute A from the corresp. triplet
        A = compute_A(u2_i, u_i)

        ux = A @ e2p(u2)
        e = np.power(u[0]-ux[0], 2) + np.power(u[1]-ux[1], 2)
        e_max = np.sqrt(np.max(e))

        if e_max < e_max_best:
            if PRINT:
                print("New best e:", e_max)
            e_max_best = e_max
            A_best = A

    return A_best


if __name__ == "__main__":
    img_arr = mpimg.imread("daliborka_01.jpg")

    # change the pixel colors
    u_coords = np.round(u).astype(int)
    img_arr[u_coords[1], u_coords[0]] = colors
    plt.imshow(img_arr)
    plt.show()
    plt.imsave("01_daliborka_points.png", img_arr, format="png")

    A = estimate_A(u2, u)
    ux = A @ e2p(u2)
    e = 100 * (ux - u)  # error displacements magnified by 100

    fig = plt.figure()  # figure handle to be used later
    fig.clf()
    plt.imshow(img_arr)

    # draw all points (in proper color) and errors
    for i in range(u.shape[1]):
        if i == 0:
            plt.plot(u[0, i], u[1, i], 'o', color=colors[i]/255, fillstyle='none', label='Points')  # the i-th point in magenta color
            plt.plot((u[0, i], u[0, i] + e[0, i]), (u[1, i], u[1, i] + e[1, i]), 'r-', label='Transf. errs.')  # the i-th displacement
        else:
            plt.plot(u[0, i], u[1, i], 'o', color=colors[i]/255, fillstyle='none')  # the i-th point in magenta color
            plt.plot((u[0, i], u[0, i] + e[0, i]), (u[1, i], u[1, i] + e[1, i]), 'r-')  # the i-th displacement

    plt.title("Points and transfer errors (100x)")
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.legend(loc='lower right')

    plt.show()
    fig.savefig('01_daliborka_errs.pdf')

    sio.savemat('01_points.mat', {'u': u, 'A': A})







