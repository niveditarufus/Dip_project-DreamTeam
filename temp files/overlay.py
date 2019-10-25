from PIL import Image
import imregpoc
import cv2
import numpy as np
import sys
import time

np.set_printoptions(threshold=sys.maxsize)


def merge_images(file1, file2,x, y,theta):
    """Merge two images into one, displayed side by side
    :param file1: path to first image file
    :param file2: path to second image file
    :return: the merged Image object
    """
    image1 = Image.open(file1)
    image2 = Image.open(file2)


    image2 = image2.rotate(-(theta*180)/np.pi, expand = "True")
    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = 10000
    result_height = 10000

    result = Image.new('L', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    # resul = Image.open("result.png")
    result.paste(im=image2, box=(int(result_width/2+x), int(result_height/2-y)), mask = image2)
    return result

# file1  = 'figure1.png'
# file2  = 'figure2.png'
# ref = cv2.imread("figure1.png",0)
# cmp = cv2.imread("figure2.png",0)
posx = 0
posy = 0
theta = 0
# ref = cv2.imread(file1,0)
# cmp = cv2.imread(file2,0)
# result = imregpoc.imregpoc(ref,cmp)
# trans = np.hypot(result.getParam()[0],result.getParam()[1])
# print(trans)
# theta = theta + result.getParam()[2]
# posy = posy + trans*np.cos(theta) 
# posx = posx + trans*np.sin(theta)
# print(posx,posy,theta)
# result = merge_images(file1,file2,posx,posy,theta)
# result.save('result.png')
result = Image.new('L', (10000, 10000))
result.save('result.png')
for i in range(1,4540):
    file1 = 'result.png'
    file2 = '/home/nive/newimages_plane/figure'+str(i+1)+'.png'
    ref = cv2.imread('/home/nive/newimages_plane/figure'+str(i)+'.png',0)
    cmp = cv2.imread('/home/nive/newimages_plane/figure'+str(i+1)+'.png',0)
    result = imregpoc.imregpoc(ref,cmp)
    trans = np.hypot(result.getParam()[0],result.getParam()[1])
    # print(trans)
    theta = theta + result.getParam()[2]
    posy = posy + trans*np.cos(theta) 
    posx = posx + trans*np.sin(theta)
    print(posx,posy,theta)
    result = merge_images(file1,file2,posx,posy,theta)
    result.save('result.png')
    if i%10 == 0:
        result.save('/home/nive/results/result'+str(i)+'.png')
    print(i)
    # time.sleep(5)