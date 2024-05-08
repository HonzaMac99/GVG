import numpy as np                # for matrix computation and linear algebra
import matplotlib.pyplot as plt   # for drawing and image I/O
from mpl_toolkits import mplot3d  # for 3d plots
import matplotlib.image as mpimg
import scipy
import scipy.linalg
import scipy.io as sio            # for matlab file format output
import itertools                  # for generating all combinations
from PIL import Image


# this script only converts png files to pdf


img = mpimg.imread("09_view3.png")

plt.imshow(img)
plt.axis("off")
plt.savefig("09_view3.pdf")
plt.show()