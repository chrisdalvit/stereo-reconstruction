import time
import numpy as np
import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import os

from reconstruction.BM import BM
from reconstruction.SGM import SGM
from reconstruction.OpenCV import OpenCV
from utils import load_images, save_point_cloud, compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, choices=['bm', 'sgm', 'opencv'])
parser.add_argument("--data", type=str, default="images/cones/")
parser.add_argument("--subpixel", action='store_true')
parser.add_argument("--kernel-size", type=int, default=15)
parser.add_argument("--max-disp", type=int, default=64)
parser.add_argument("--P1", type=int, default=10)
parser.add_argument("--P2", type=int, default=120)
parser.add_argument("--language", type=str, default='python', choices=['python', 'c'])
args = parser.parse_args()

left, right, gt = load_images(args.data)

if args.method == 'bm':
    matcher = BM(args.kernel_size, args.max_disp, args.subpixel, args.language)
elif args.method == 'sgm':
    matcher = SGM(args.kernel_size, args.max_disp, args.P1, args.P2, args.subpixel)
elif args.method == 'opencv':
    matcher = OpenCV(args.kernel_size, args.max_disp)
    
start = time.time()
disparity = matcher.compute(np.array(left.convert('L')), np.array(right.convert('L')))
end = time.time()

disparity = cv.medianBlur(disparity, 5)
colors = np.array(left)
print(f"Computation time: {end-start:.2f}s")
print(f"Metrics: {compute_metrics(disparity, np.array(gt))}")

if not os.path.exists("output"):
    os.mkdir("output")
save_point_cloud(f"output/{args.method}_{args.language}.ply", disparity, colors)
plt.imsave(f"output/{args.method}_{args.language}.png", disparity, cmap='jet')
