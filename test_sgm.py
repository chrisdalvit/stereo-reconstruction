from utils import load_images
from reconstruction.SGM import SGM
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import ctypes

left, right, gt = load_images("cones/")
sgm = SGM(19, 56, 20, 120, False)
gt_census = sgm._census_transform(np.array(left.convert('L')))

# --------------------------------------------------------------
left = np.double(left)
right = np.double(right)
output = np.double(np.zeros_like(left))
# Load the shared library into ctypes
libname = pathlib.Path().absolute() / "c/sgm.so"
c_lib = ctypes.CDLL(libname)
# Define the argument and return types for the function
c_lib.sgm.argtypes = [
    ctypes.POINTER(ctypes.c_double), # left
    ctypes.POINTER(ctypes.c_double), # right
    ctypes.POINTER(ctypes.c_double), # output
    ctypes.c_int, # height
    ctypes.c_int, # width
    ctypes.c_int, # max disparity
    ctypes.c_int # kernel_size
]
c_lib.sgm.restype = ctypes.c_void_p
left_ctypes = left.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
right_ctypes = right.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
out_ctypes = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
height = ctypes.c_int(left.shape[0])
width = ctypes.c_int(left.shape[1])
max_disparity = ctypes.c_int(56)
kernel_size = ctypes.c_int(19)
c_lib.sgm(left_ctypes, right_ctypes, out_ctypes, height, width, max_disparity, kernel_size)
output = np.ctypeslib.as_array(out_ctypes, shape=(left.shape[0], left.shape[1]))

plt.imsave("test_sgm_c.png", np.float32(output), cmap="gray")

plt.imsave("test_sgm_gt.png", gt_census, cmap="gray")