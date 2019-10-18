import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
#importing images
img1 = cv2.imread('./input images/figure1.png',0)
cx,cy = img1.shape
#generate a hanning window
hanw = cv2.createHanningWindow((cx,cy),cv2.CV_64F)



img1 = img1 * hanw
#Do fft
f1 = np.fft.fft2(img1)
fshift1 = np.fft.fftshift(f1)
magnitude_spectrum1 = np.abs(fshift1)
log_magnitude_spectrum1 = 10*np.log(np.abs(fshift1) +1)
magnitude_spectrum1 = 15*np.log(np.abs(fshift1) +1)
polar_map1= cv2.linearPolar(magnitude_spectrum1, (cx/2,cy/2), cx/np.log(cx), flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
log_polar_map1= cv2.linearPolar(log_magnitude_spectrum1, (cx/2,cy/2), cx/np.log(cx), flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
cv2.imwrite('./output/log_mag1.png',log_magnitude_spectrum1)
cv2.imwrite('./output/mag1.png',magnitude_spectrum1)
cv2.imwrite('./output/log_polar1.png',log_polar_map1)
cv2.imwrite('./output/polar_map1.png',polar_map1)



#read the image and find polar map using the linearPolar function to the magnitude spectrum



R = fshift1 * np.ma.conjugate(fshift2)
R /= np.absolute(R)
r = np.fft.fftshift(np.fft.ifft(R).real)
r = np.asarray(r)
r=r.reshape(cx,cy)
