from __future__ import division
import cv2, scipy, numpy, math
from matplotlib import pyplot
from PIL import Image
from pylab import *
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d

im1 = cv2.imread("/Users/virajj/Downloads/basketball2.png", 0)
im2 = cv2.imread("/Users/virajj/Downloads/basketball2.png", 0)

Gi1 = gaussian_filter(im1, 1)
Gi2 = gaussian_filter(im2, 1)

Ix = convolve2d(Gi1, [[-0.25, 0.25], [-0.25, 0.25]])
Iy = convolve2d(Gi1, [[-0.25, -0.25], [0.25, 0.25]])
#It = convolve2d(Gi1, [[0.25, 0.25], [0.25, 0.25]], 'same') + convolve2d(Gi2, [[-0.25, -0.25], [-0.25, -0.25]], 'same')
It = convolve2d(im1, 0.25 * np.ones((2,2))) + convolve2d(im2, -0.25 * np.ones((2,2)))
cv2.imshow("Ix", Ix)
cv2.waitKey(0)
cv2.imshow("Iy", Iy)
cv2.waitKey(0)
cv2.imshow("It", It)
cv2.waitKey(0)

cv2.destroyAllWindows()
