import numpy as np
import cv2 as cv

def save_point_cloud(filename, disparity, colors):
    Q = np.array([[1, 0, 0, -disparity.shape[1]/2],
                [0, -1, 0, disparity.shape[0]/2],
                [0, 0, 0, -0.8*disparity.shape[1]],  # focal length; adjust based on calibration
                [0, 0, 1/0.05, 0]])  # baseline (adjust based on your setup)

    points_3d = cv.reprojectImageTo3D(disparity, Q)
    mask = disparity > 0  # Remove points where disparity is invalid
    points = points_3d[mask]
    colors = colors[mask]
    points = np.hstack([points, colors])
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, "w") as f:
        f.write(header)
        np.savetxt(f, points, fmt="%f %f %f %d %d %d")