import cv2 as cv
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Step 1: Load Stereo Images
root = "cones/"
left_img = cv.imread(root+"left.png", cv.IMREAD_GRAYSCALE)
right_img = cv.imread(root+"right.png", cv.IMREAD_GRAYSCALE)
print(left_img.shape)

stereo = cv.StereoBM_create(blockSize=5)
disparity = stereo.compute(left_img,right_img)
plt.imshow(disparity,'gray')
plt.show()