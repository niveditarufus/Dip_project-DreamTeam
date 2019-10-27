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
# Basedir where the dataset exsits
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
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
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
	dataset = pykitti.odometry(basedir,sequence)	 # Reads KITTI odometry dataset
	num_frames = len(dataset.poses) # Number of frames in the particular sequence 
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

		# print(fit.shape)
		# errors = b - (A.dot(np.transpose(fit[i])))
		# residual = np.linalg.norm(errors)
		# print(fit[i,0])

		if(i>0):
			theta_plane = get_theta_between_planes(fit[i-1,0],fit[i,0],fit[i-1,1],fit[i,1])

		projected =[]
		perpendicular_distance=[]
		# for element in velo_xy:
		# 	perpendicular_distance.append(shortest_distance(element[0],element[1],element[2],fit[0],fit[1],fit[2]))
		# perpendicular_distance = np.asarray(perpendicular_distance)
		# perpendicular_distance = np.clip(a=perpendicular_distance,a_min=-1.6,a_max=1.6)
		pixel_values = scale_to_255(velo_xy[:,2],min=height_range[0],max=height_range[1])
		for element in velo_xy:
			projected.append(projection(fit[i,0],fit[i,1],fit[i,2],element[0],element[1],element[2]))

		projected = np.asarray(projected)

		# fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
		# mayavi.mlab.points3d(projected[:,0], projected[:,1], projected[:,2],mode="point",colormap='spectral', # 'bone', 'copper', 'gnuplot'# color=(0, 1, 0),
		# 					figure=fig,)
		# mayavi.mlab.show()
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


		
		# mayavi.mlab.show()
		# plt.figure()
		# ax = plt.subplot(111, projection='3d')
		# ax.scatter(projected[:,0], projected[:,1], projected[:,2], color='b')

		# xlim = ax.get_xlim()
		# ylim = ax.get_ylim()
		# X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
		#                   np.arange(ylim[0], ylim[1]))
		# Z = np.zeros(X.shape)
		# for r in range(X.shape[0]):
		#     for c in range(X.shape[1]):
		#         Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
		# ax.plot_wireframe(X,Y,Z, color='k')

		# ax.set_xlabel('x')
		# ax.set_ylabel('y')
		# ax.set_zlabel('z')
		# plt.show()
		# for i in range(iterations):
		# 	data_index = [np.random.randint(0, velo_filt.shape[0], size=3)]
		# 	data = np.array(velo_filt[tuple(data_index)])
		# 	print(data.shape)
		# 	a,b,c,d = equation_plane(data[0,0],data[0,1],data[0,2],data[1,0],data[1,1],data[1,2],data[2,0],data[2,1],data[2,2])
		# 	for element in velo_filt
		# 	check_plane(a, b, c, d, element[0], element[1], element[2])

		 # Read the i th frame from the sequence
		# fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
		# mayavi.mlab.points3d(velo[:,0], velo[:,1], velo[:,2],mode="point",colormap='spectral', # 'bone', 'copper', 'gnuplot'# color=(0, 1, 0),
		# 					figure=fig,)
		                         
		# mayavi.mlab.show()
		# pointcloud = np.fromfile(str("000010.bin"), dtype=np.float32, count=-1).reshape([-1,4])
		# im = top.point_cloud_2_birdseye(velo)
		# im2 = Image.fromarray(im)
		# im2.save('/home/nive/newimages/figure'+str(i)+'.png')
		# print(i, velo_filt.shape, velo.shape)