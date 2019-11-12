import cv2
from PIL import Image
import numpy as np
img1 = cv2.imread('./input images/figure1.png',0)
cx,cy = img1.shape
hanw = cv2.createHanningWindow((cx,cy),cv2.CV_64F)

img1 = img1 * hanw
f1 = np.fft.fft2(img1)
fshift1 = np.fft.fftshift(f1)
magnitude_spectrum1 = np.log(np.abs(fshift1) +1)
polar_map1= cv2.logPolar(magnitude_spectrum1, (cx/2,cy/2), cx/np.log(cx), flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)

img2 = cv2.imread('./input images/figure2.png',0)
img2 = img2 * hanw
f2 = np.fft.fft2(img2)
fshift2 = np.fft.fftshift(f2)
magnitude_spectrum2 = np.log(np.abs(fshift2) +1)
polar_map2= cv2.logPolar(magnitude_spectrum2, (cx/2,cy/2), cx/np.log(cx), flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)

R = fshift1 * np.ma.conjugate(fshift2)
R /= np.absolute(R)
r = np.fft.fftshift(np.fft.ifft(R).real)
r = np.asarray(r)
r=r.reshape(cx,cy)
DY,DX = np.unravel_index(r.argmax(), r.shape)

