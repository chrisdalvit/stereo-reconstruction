import cv2 as cv
import matplotlib.pyplot as plt
from point_cloud import save_point_cloud

# Step 1: Load Stereo Images
root = "cones/"
left_img = cv.imread(root+"left.png", cv.IMREAD_GRAYSCALE)
right_img = cv.imread(root+"right.png", cv.IMREAD_GRAYSCALE)
print(left_img.shape)

stereo = cv.StereoBM_create(blockSize=11)
disparity = stereo.compute(left_img,right_img)
disp_map = cv.medianBlur(disparity, 5)
colors = cv.cvtColor(cv.imread(root + "/left.png"), cv.COLOR_BGR2RGB)
save_point_cloud(f"cv.ply", disparity, colors)
plt.imshow(disparity,'gray')
plt.show()