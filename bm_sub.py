from tqdm import tqdm
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv

from point_cloud import save_point_cloud

BLOCK_SIZE = 11
SEARCH_BLOCK_SIZE = 56

def get_patch(y, x, img, kernel_half, offset=0):
    return img[y-kernel_half:y+kernel_half, x-kernel_half-offset+1:x+kernel_half-offset+1]

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--data", type=str, default="cones/")
args = parser.parse_args()

left_img = np.array(Image.open(args.data + "/left.png").convert('L'))
right_img = np.array(Image.open(args.data + "/right.png").convert('L'))
h, w = left_img.shape
    
disp_map = np.full_like(left_img, -1, dtype=np.float32)
kernel_half = int(BLOCK_SIZE / 2)
offset_adjust = 255 / SEARCH_BLOCK_SIZE  # this is used to map depth map output to 0-255 range

for y in tqdm(range(kernel_half, h - kernel_half)):      
    for x in range(kernel_half, w - kernel_half):
        best_offset = None
        min_error = float("inf")
        errors = []
        for offset in range(SEARCH_BLOCK_SIZE):               
            left_patch = get_patch(y, x, left_img, kernel_half)
            right_patch = get_patch(y, x, right_img, kernel_half, offset)
            if left_patch.shape != right_patch.shape:
                errors.append(None)
                continue
            error = np.sum((left_patch - right_patch)**2)
            errors.append(np.float32(error))
            if error < min_error:
                min_error = error
                best_offset = offset
        
                # Subpixel interpolation
        if 1 <= best_offset < SEARCH_BLOCK_SIZE - 1 and errors[best_offset - 1] and errors[best_offset + 1]:
            denom = errors[best_offset - 1] + errors[best_offset + 1] - 2 * errors[best_offset]
            if denom != 0:
                subpixel_offset = (errors[best_offset - 1] - errors[best_offset + 1]) / (2 * denom)
                best_offset = best_offset + subpixel_offset
        
        disp_map[y, x] = best_offset * offset_adjust

disp_map = cv.medianBlur(disp_map, 5)[:,SEARCH_BLOCK_SIZE:]
colors = cv.cvtColor(cv.imread(args.data + "/left.png")[:,SEARCH_BLOCK_SIZE:], cv.COLOR_BGR2RGB)
save_point_cloud(f"{args.name}.ply", disp_map, colors)
plt.imshow(disp_map, cmap='jet')
plt.savefig(f"{args.name}_plot.png")
plt.show()