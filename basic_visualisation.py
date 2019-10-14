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
magnitude_spectrum1 = 15*np.log(np.abs(fshift1) +1)
polar_map1= cv2.linearPolar(magnitude_spectrum1, (cx/2,cy/2), cx/np.log(cx), flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
cv2.imwrite('./output/mag1.png',magnitude_spectrum1)


img2 = cv2.imread('./input images/figure2.png',0)
img2 = img2 * hanw
f2 = np.fft.fft2(img2)
fshift2 = np.fft.fftshift(f2)
magnitude_spectrum2 =15* np.log(np.abs(fshift2) +1)
polar_map2= cv2.linearPolar(magnitude_spectrum2, (cx/2,cy/2), cx/np.log(cx), flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)
cv2.imwrite('./output/mag2.png',magnitude_spectrum2)

R = fshift1 * np.ma.conjugate(fshift2)
R /= np.absolute(R)
r = np.fft.fftshift(np.fft.ifft(R).real)
r = np.asarray(r)
r=r.reshape(cx,cy)
