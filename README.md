# Dip_project_DreamTeam 
The objective of the project is to estimate the rotation of two images from the polar mapped amplitude spectra. The proposed method is to estimate the translational shift in the transformed coordinate system by employing 1D phase correlation with adaptive line selection which will reduce the computation cost by 50%. An unofficial implementation of the paper: 

https://link.springer.com/content/pdf/10.1007%2F978-3-540-74260-9_19.pdf?fbclid=IwAR3pWfo8b5DyoIsM4Oy8f6cWqvzhgJ3JfMg6ZBl2aGXp-xP_BYaH75Yb7vk 

Nagashima, Sei & Ito, Koichi & Aoki, Takafumi & Ishii, Hideaki & Kobayashi, Koji. (2009). High-Accuracy Estimation of Image Rotation Using 1D Phase-Only Correlation. IEICE Transactions. 92-A. 235-243. 10.1587/transfun.E92.A.235.

Directory structure:

Files:
basic_visualisation.py : This file takes input from 'input images'folder and writes the output in 'output' folder
dip_project.ipynb : This file implements the paper given in the link.

Folders:
input images : This folder contains all the input images used in the code. Any images can that required to be tested should be put in this folder.
output : This folder contains outputs on executing th file 'basic_visualisation.py'
rotated : This folder contains rotated versions of images present in 'input images' folder.

*All outputs can be visualised in the jupyter notebook(dip_project.ipynb) itself. Clear descriptions are made in respective markdowns. These descriptions are taken from the paper cited above*

Python libraries required:
OpenCV
Numpy
Matplotlib
Scipy
