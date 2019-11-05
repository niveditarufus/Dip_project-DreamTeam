import numpy as np
import top
from PIL import Image
import pykitti
import mayavi.mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import sys

np.set_printoptions(threshold=sys.maxsize)
res = 0.1
side_range=(-50., 50.)
fwd_range = (-50., 50.)
basedir = '/home/nive/space' 

sequences = ['00']#,'01','02','03','04','05','06','07','08','09','10']
height_range_plane = [-1.80, -1.60]
height_range = [-1.6, 1]
iterations = 1

def get_theta_between_planes(a1,a2,b1,b2):
	sq1 = np.sqrt((a1*a1) + (b1*b1) + (1))
	sq2 = np.sqrt((a2*a2) + (b2*b2) + (1))
	d = (a2*a1) + (b2*b1) + (1)
	return(math.acos(d/(sq1*sq2)))


def scale_to_255(a, min, max, dtype=np.uint8):
    
    return (((a - min) / float(max - min)) * 255).astype(dtype)

def shortest_distance(x1, y1, z1, a, b, c):  
      
    distance = abs((a * x1 + b * y1 + -1* z1 + c))  
    e = (np.sqrt(a * a + b * b + 1))
    return distance/e


def projection(a,b,c,x,y,z):
	unit_normal = np.sqrt(a*a + b*b + 1)
	dist = ((a*x) + (b*y) +(-1*z))/unit_normal + c
	projected_x = x-(dist*a/unit_normal)
	projected_y = y-(dist*b/unit_normal)
	projected_z = z-(dist*-1/unit_normal)
	return (projected_x, projected_y, projected_z)


for sequence in sequences:	
	dataset = pykitti.odometry(basedir,sequence)	 
	num_frames = len(dataset.poses) 
	print(num_frames) 

	for i in range(num_frames):
		velo = dataset.get_velo(i)
		x_filt = np.logical_and((velo[:,0] > fwd_range[0]), (velo[:,0] < fwd_range[1]))
		y_filt = np.logical_and((velo[:,1] > -side_range[1]), (velo[:,1]< -side_range[0]))
		z_filt = np.logical_and((velo[:,2] > height_range_plane[0]), (velo[:,2] < height_range_plane[1]))
		z_height_filt = np.logical_and((velo[:,2] > height_range[0]), (velo[:,2] < height_range[1]))
		filt_temp = np.logical_and(x_filt,y_filt)
		filt_project = np.logical_and(filt_temp,z_height_filt)
		xy_indices = np.argwhere(filt_project).flatten()
		velo_xy = np.asarray(velo[xy_indices])
		filt = np.logical_and(filt_temp,z_filt)
		indices = np.argwhere(filt).flatten()
		velo_filt = np.asarray(velo[indices])
		tmp_A =[]
		tmp_b = []
		for index in range(velo_filt.shape[0]):
			tmp_A.append([velo_filt[index,0], velo_filt[index,1], 1])
			tmp_b.append(velo_filt[index,2])
		b = np.transpose(np.asarray(tmp_b))
		A = np.asarray(tmp_A)
		temp = np.linalg.inv(np.transpose(A).dot(A))
		temp1 = temp.dot(np.transpose(A))
		fit=np.empty((A.shape[0],3))
		theta_plane=np.empty(num_frames-1)
		fit[i] = np.array([temp1.dot(b)])

		
		if(i>0):
			theta_plane = get_theta_between_planes(fit[i-1,0],fit[i,0],fit[i-1,1],fit[i,1])

		projected =[]
		perpendicular_distance=[]
		
		pixel_values = scale_to_255(velo_xy[:,2],min=height_range[0],max=height_range[1])
		for element in velo_xy:
			projected.append(projection(fit[i,0],fit[i,1],fit[i,2],element[0],element[1],element[2]))

		projected = np.asarray(projected)

		x_img = (-projected[:,1] / res).astype(np.int32)  # x axis is -y in LIDAR
		y_img = (-projected[:,0] / res).astype(np.int32)
		x_img -= int(np.floor(side_range[0] / res))
		y_img += int(np.ceil(fwd_range[1] / res))
		x_max = 1 + int((side_range[1] - side_range[0]) / res)
		y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
		im = np.zeros([x_max,y_max], dtype=np.uint8)
		im[y_img,x_img] = pixel_values
		im2 = Image.fromarray(im)
		im2.save('/home/nive/newimages_plane/figure'+str(i)+'.png')
		print(i)

		