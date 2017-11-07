from __future__ import division
import cv2, scipy, numpy, math
from matplotlib import pyplot
from PIL import Image
from pylab import *
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d

im1 = cv2.imread("/Users/virajj/Downloads/basketball1.png", 0) #read image 1
im2 = cv2.imread("/Users/virajj/Downloads/basketball2.png", 0) #read image 2

Gi1 = gaussian_filter(im1, 1)
Gi2 = gaussian_filter(im2, 1)

feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.2,
                       minDistance = 1,
                       blockSize = 7 )

corners = np.int0(cv2.goodFeaturesToTrack(Gi1, mask=None, **feature_params)) # get corners
im1_c = cv2.cvtColor(Gi1, cv2.COLOR_GRAY2RGB)
for c in corners:
    i, j = c.ravel()
    cv2.circle(im1_c,(i,j), 1 ,255,-1) #highlight corners with blue


Ix = convolve2d(im1, [[-0.25, 0.25], [-0.25, 0.25]]) + convolve2d(im2, [[-0.25, 0.25], [-0.25, 0.25]]) #find f_x:
Iy = convolve2d(im1, [[-0.25, -0.25], [0.25, 0.25]]) + convolve2d(im2, [[-0.25, -0.25], [0.25, 0.25]]) # find f_y:
#f_t = convolve2d(Gi1, [[0.25, 0.25], [0.25, 0.25]], 'same') + convolve2d(Gi2, [[-0.25, -0.25], [-0.25, -0.25]], 'same')
It = convolve2d(im1, 0.25 * np.ones((2,2))) + convolve2d(im2, -0.25 * np.ones((2,2))) # find f_t

#for i in range(10, im1.shape[0]-5, 10):
#    for j in range(10, im1.shape[1]-5, 10):
for c in corners:
    j, i = c.ravel()
    fx = [Ix[i-1][j-1], Ix[i-1][j], Ix[i-1][j+1], Ix[i][j-1], Ix[i][j], Ix[i][j+1], Ix[i+1][j-1], Ix[i+1][j], Ix[i+1][j+1]]#get neighboring pixels 3X3
    fy = [Iy[i-1][j-1], Iy[i-1][j], Iy[i-1][j+1], Iy[i][j-1], Iy[i][j], Iy[i][j+1], Iy[i+1][j-1], Iy[i+1][j], Iy[i+1][j+1]]#get neighboring pixels 3X3
    ft = [It[i-1][j-1], It[i-1][j], It[i-1][j+1], It[i][j-1], It[i][j], It[i][j+1], It[i+1][j-1], It[i+1][j], It[i+1][j+1]]#get neighboring pixels 3X3
    A = numpy.hstack([numpy.vstack(fx), numpy.vstack(fy)])#get A = [fx fy], a 2X9 matrix
    F = - numpy.vstack(ft)#get F = [-ft]: 1X9 matrix
    u, v =  numpy.dot(numpy.linalg.pinv(A), F) # get u, v:- displacement in x and y direction respectively
    #numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(A.T,A)),A.T), F)
    #print u, "\n\n"
    cv2.arrowedLine(im1_c, (i, j), (int(math.ceil(i+u)), int(math.ceil(j+v))), (0,0,255), 2)# draw displacement vectors
#    print(u, v)

cv2.imshow("Ix", Ix)
cv2.waitKey(0)
cv2.imshow("Iy", Iy)
cv2.waitKey(0)
cv2.imshow("It", It)
cv2.waitKey(0)
cv2.imshow("res", im1_c)
cv2.waitKey(0)
cv2.destroyAllWindows()

