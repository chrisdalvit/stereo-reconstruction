import time
import numpy as np
import argparse
import cv2 as cv
import matplotlib.pyplot as plt

from reconstruction.BM import BM
from reconstruction.SGM import SGM
from utils import load_images, save_point_cloud, compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, choices=['bm', 'sgm', 'opencv'])
parser.add_argument("--data", type=str, default="cones/")
parser.add_argument("--subpixel", action='store_true')
parser.add_argument("--kernel-size", type=int, default=19)
parser.add_argument("--max-disp", type=int, default=64)
parser.add_argument("--P1", type=int, default=10)
parser.add_argument("--P2", type=int, default=120)
args = parser.parse_args()

left, right, gt = load_images(args.data)

if args.method == 'bm':
    matcher = BM(args.kernel_size, args.max_disp, args.subpixel)
elif args.method == 'sgm':
    matcher = SGM(args.kernel_size, args.max_disp, args.P1, args.P2, args.subpixel)
elif args.method == 'opencv':
    matcher = cv.StereoBM_create(numDisparities=args.max_disp, blockSize=args.kernel_size)
    matcher.setPreFilterType(cv.STEREO_BM_PREFILTER_XSOBEL)
    
start = time.time()
disparity = matcher.compute(np.array(left.convert('L')), np.array(right.convert('L')))
end = time.time()
if args.method == 'opencv':
    disparity = disparity.astype(np.float32) / 4.0
disparity = cv.medianBlur(disparity, 5)
colors = np.array(left)
print(f"Computation time: {end-start:.2f}s")
print(f"Metrics: {compute_metrics(np.array(gt), disparity)}")

save_point_cloud(f"{args.method}.ply", disparity, colors)
plt.savefig(f"{args.method}.png")
