import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from point_cloud import save_point_cloud

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="sgm")
parser.add_argument("--data", type=str, default="cones/")
args = parser.parse_args()

BLOCK_SIZE = 19
SEARCH_BLOCK_SIZE = 56

left_img = np.array(Image.open(args.data + "/left.png").convert('L'))
right_img = np.array(Image.open(args.data + "/right.png").convert('L'))

def get_patch(y, x, img, kernel_half, offset=0):
    return img[y-kernel_half:y+kernel_half+1, x-kernel_half-offset:x+kernel_half-offset+1]

def census_transform(img, kernel_size):
    height, width = img.shape
    kernel_half = kernel_size // 2
    census_values = np.zeros(shape=(height, width), dtype=np.int32)
    for y in range(kernel_half, height - kernel_half):
        for x in range(kernel_half, width - kernel_half):
            patch = get_patch(y, x, img, kernel_half)
            # If value is less than center value assign 1 otherwise assign 0 
            census_pixel_array = (patch.flatten() > img[y, x]).astype(int)
            # Convert census array to an integer by using bit shift operator
            census_values[y, x] = np.int32(census_pixel_array.dot(1 << np.arange(kernel_size * kernel_size)[::-1])) 
    return census_values

def compute_costs(left_census_values, right_census_values, max_disparity, kernel_half):
    height, width = left_census_values.shape
    
    left_cost_volume = np.zeros(shape=(height, width, max_disparity), dtype=np.uint32)
    right_cost_volume = np.zeros(shape=(height, width, max_disparity), dtype=np.uint32)
    left_tmp = np.zeros_like(left_census_values, dtype=np.int32)
    right_tmp = np.zeros_like(left_census_values, dtype=np.int32)

    for d in range(max_disparity):
        # The right image is shifted d pixels accross
        right_tmp[:, kernel_half+d:width-kernel_half] = right_census_values[:, kernel_half:width-d-kernel_half]
        # 1 is assigned when the bits differ and 0 when they are the same
        left_xor = np.bitwise_xor(left_census_values, right_tmp)
        # All the 1's are summed up to give us the number of different pixels (the cost)
        left_distance = np.zeros_like(left_census_values, dtype=np.uint32)
        while not np.all(left_xor == 0):
            tmp = left_xor - 1
            mask = left_xor != 0
            left_xor[mask] &= tmp[mask]
            left_distance[mask] += 1
        # All the costs for that disparity are added to the cost volume
        left_cost_volume[:, :, d] = left_distance

        left_tmp[:, kernel_half:width-d-kernel_half] = left_census_values[:, kernel_half+d:width-kernel_half]
        right_xor = np.bitwise_xor(right_census_values, left_tmp)
        right_distance = np.zeros_like(right_census_values, dtype=np.uint32)
        while not np.all(right_xor == 0):
            tmp = right_xor - 1
            mask = right_xor != 0
            right_xor[mask] &= tmp[mask]
            right_distance[mask] += 1
        right_cost_volume[:, :, d] = right_distance
    return left_cost_volume, right_cost_volume

# --------------------------------------------------------------------
def get_penalties(max_disparity, P2, P1):
    p2 = np.full(shape=(max_disparity, max_disparity), fill_value=P2, dtype=np.int32)
    p1 = np.full(shape=(max_disparity, max_disparity), fill_value=P1 - P2, dtype=np.int32)
    p1 = np.tril(p1, k=1) # keep values lower than k'th diagonal
    p1 = np.triu(p1, k=-1) # keep values higher than k'th diagonal
    no_penalty = np.identity(max_disparity, dtype=np.int32) * -P1 # create diagonal matrix with values -p1
    penalties = p1 + p2 + no_penalty
    return penalties

def get_path_cost(slice, offset, penalties, other_dim, disparity_dim):
    minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=np.int32)
    minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

    for pixel_index in range(offset, other_dim):
        # Get all the minimum disparities costs from the previous pixel in the path
        previous_cost = minimum_cost_path[pixel_index - 1, :]
        # Get all the disparities costs (from the cost volume) for the current pixel
        current_cost = slice[pixel_index, :]
        costs = np.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
        # Add penalties to the previous pixels disparities that differ from current pixels disparities
        costs = costs + penalties
        # Find minimum costs for the current pixels disparities using the previous disparities costs + penalties 
        costs = np.amin(costs, axis=0)  
        # Current pixels disparities costs + minimum previous pixel disparities costs (with penalty) - 
        # (constant term) minimum previous cost from all disparities 
        pixel_direction_costs = current_cost + costs - np.amin(previous_cost)
        minimum_cost_path[pixel_index, :] = pixel_direction_costs

    return minimum_cost_path    


def aggregate_costs(cost_volume, P2, P1, height, width, disparities):
    penalties = get_penalties(disparities, P2, P1)

    south_aggregation = np.zeros(shape=(height, width, disparities), dtype=np.float32)
    north_aggregation = np.copy(south_aggregation)

    for x in range(0, width):
        # Takes all the rows and disparities for a single column
        south = cost_volume[:, x, :]
        # Invert the rows to get the opposite direction
        north = np.flip(south, axis=0)
        south_aggregation[:, x, :] = get_path_cost(south, 1, penalties, height, disparities)
        north_aggregation[:, x, :] = np.flip(get_path_cost(north, 1, penalties, height, disparities), axis=0)

    east_aggregation = np.copy(south_aggregation)
    west_aggregation = np.copy(south_aggregation)
    for y in range(0, height):
        # Takes all the column and disparities for a single row
        east = cost_volume[y, :, :]
        # Invert the columns to get the opposite direction
        west = np.flip(east, axis=0)
        east_aggregation[y, :, :] = get_path_cost(east, 1, penalties, width, disparities)
        west_aggregation[y, :, :] = np.flip(get_path_cost(west, 1, penalties, width, disparities), axis=0)

    # Combine the costs from all paths into a single aggregation volume
    aggregation_volume = np.concatenate((south_aggregation[..., None], north_aggregation[..., None], east_aggregation[..., None], west_aggregation[..., None]), axis=3)
    return aggregation_volume

def select_disparity(aggregation_volume):
    # sum up costs for all directions
    volume = np.sum(aggregation_volume, axis=3).astype(float)
    
    # returns the disparity index with the minimum cost associated with each h x w pixel
    disparity_map = np.argmin(volume, axis=2).astype(float)
    h, w, c = volume.shape
    for i in range(h):
        for j in range(w):
            d = int(disparity_map[i,j])
            if 0 < d < c-1:
                denom = volume[i,j,d-1] + volume[i,j,d+1] - 2 * volume[i,j,d]
                if denom != 0:
                    subpixel_offset = (volume[i,j,d-1] - volume[i,j,d+1]) / (2 * denom)
                    disparity_map[i,j] += subpixel_offset
    return disparity_map
# --------------------------------------------------------------------

P1 = 10 # penalty for disparity difference = 1
P2 = 120 # penalty for disparity difference > 1
kernel_half = BLOCK_SIZE // 2
h, w = left_img.shape
offset_adjust = 255 / SEARCH_BLOCK_SIZE  # this is used to map depth map output to 0-255 range

left_census = census_transform(left_img, BLOCK_SIZE)
right_census = census_transform(right_img, BLOCK_SIZE)
left_cost_volume, right_cost_volume = compute_costs(left_census, right_census, SEARCH_BLOCK_SIZE, kernel_half)
left_aggregation_volume = aggregate_costs(left_cost_volume, P2, P1, h, w, SEARCH_BLOCK_SIZE)
right_aggregation_volume = aggregate_costs(right_cost_volume, P2, P1, h, w, SEARCH_BLOCK_SIZE)

left_disparity_map = np.float32(255*select_disparity(left_aggregation_volume) / SEARCH_BLOCK_SIZE)
right_disparity_map = np.float32(255*select_disparity(right_aggregation_volume) / SEARCH_BLOCK_SIZE)

left_disparity_map = cv.medianBlur(left_disparity_map, 5)[:,SEARCH_BLOCK_SIZE:]
colors = cv.cvtColor(cv.imread(args.data + "/left.png")[:,SEARCH_BLOCK_SIZE:], cv.COLOR_BGR2RGB)
save_point_cloud(f"{args.name}.ply", left_disparity_map, colors)
plt.imshow(left_disparity_map, cmap='jet')
plt.savefig(f"{args.name}_plot.png")
plt.show()