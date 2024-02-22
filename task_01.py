import numpy as np
import matplotlib.pyplot as plt
import mpimg


img = mpimg.imread("daliborka_01.jpg")
# img = np.load("daliborka_01.jpg")

U2 = np.array([[ -182.6,  -170.5,  -178.8,  -202.6,    51.5,   -78.0,   106.1],
               [  265.8,   447.0,   486.7,   851.9,   907.1,  1098.7,  1343.6]])

plt.show()