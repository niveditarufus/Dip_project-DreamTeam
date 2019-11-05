
# to find the adaptive filtering
import cv2  
import numpy as np  
# path to input image is specified and   
# image is loaded with imread command  
image1 = cv2.imread('input1.jpg')  
# cv2.cvtColor is applied over the 
# image input with applied parameters 
# to convert the image in grayscale  
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) 
# applying different thresholding  
# techniques on the input image 
thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                          cv2.THRESH_BINARY, 199, 5) 
  
thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 199, 5) 
# the window showing output images 
# with the corresponding thresholding  
# techniques applied to the input image 
cv2.imshow('Adaptive Mean', thresh1) 
cv2.imshow('Adaptive Gaussian', thresh2) 
