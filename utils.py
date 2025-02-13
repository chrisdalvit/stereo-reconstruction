import numpy as np
from PIL import Image
import cv2 as cv

def load_images(root):
    "Load left, right and ground truth images"
    left = Image.open(root + "/left.png")
    right = Image.open(root  + "/right.png")
    gt = Image.open(root  + "/gt.png")
    return left, right, gt

def save_point_cloud(filename, disparity, colors):
    """Save a 3D point cloud to a PLY file"""
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
        

def compute_metrics(pred, gt, threshold=3.0):
    """Compute error metrics between predicted and ground truth disparity maps"""
    mask = gt > 0  # Consider only valid disparity values
    valid_pred = pred[mask].astype(np.float32)
    valid_gt = gt[mask].astype(np.float32)
    
    abs_error = np.abs(valid_pred - valid_gt)
    epe = np.mean(abs_error)
    bad_pixels = np.sum(abs_error > threshold) / len(valid_gt) * 100
    rmse = np.sqrt(np.mean((valid_pred - valid_gt) ** 2))
    mae = np.mean(abs_error)

    return {
        "EPE": round(float(epe), 2),
        "Bad Pixel %": round(float(bad_pixels), 2),
        "RMSE": round(float(rmse), 2),
        "MAE": round(float(mae), 2)
    }