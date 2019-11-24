import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
#importing images
img1 = cv2.imread('./input images/facebook.png',0)
cx,cy = img1.shape
#generate a hanning window
hanw = cv2.createHanningWindow((cx,cy),cv2.CV_64F)
#read the image and find polar map using the logPolar function to the magnitude spectrum

img1 = img1 * hanw
#Do fft
f1 = np.fft.fft2(img1)
fshift1 = np.fft.fftshift(f1)
magnitude_spectrum1 = np.abs(fshift1)
log_magnitude_spectrum1 = 10*np.log(np.abs(fshift1) +1)
magnitude_spectrum1 = 15*np.log(np.abs(fshift1) +1)
# polar map of the magnitude spectrum
polar_map1= cv2.linearPolar(magnitude_spectrum1, (cx/2,cy/2), cx/np.log(cx), flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
#logarithm for ease of classifcation
M90 = cv2.getRotationMatrix2D((cx/2,cy/2), 90, 1)
polar_map1 = cv2.warpAffine(polar_map1, M90, (cx, cy))
# writing all these files
cv2.imwrite('./output/log_mag1.png',log_magnitude_spectrum1)
cv2.imwrite('./output/mag1.png',magnitude_spectrum1)
cv2.imwrite('./output/polar_map1.png',polar_map1)