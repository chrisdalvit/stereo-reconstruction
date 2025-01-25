import ctypes
import pathlib
import numpy as np
import time
import cv2 as cv

from utils import load_images
import matplotlib.pyplot as plt

if __name__ == "__main__":
    left, right, gt = load_images("cones/")
    
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

    left_arr = np.array(left.convert('L'), dtype=np.double)
    right_arr = np.array(right.convert('L'), dtype=np.double)
    output = np.double(np.ones_like(left_arr))
    
    left_ctypes = left_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    right_ctypes = right_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    out_ctypes = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    height = ctypes.c_int(left_arr.shape[0])
    width = ctypes.c_int(left_arr.shape[1])
    max_disparity = ctypes.c_int(56)
    kernel_size = ctypes.c_int(11)
    
    start = time.time()
    c_lib.sgm(left_ctypes, right_ctypes, out_ctypes, height, width, max_disparity, kernel_size)
    end = time.time()
    print(f"Time {end-start}s")
    
    output = np.ctypeslib.as_array(out_ctypes, shape=(left_arr.shape[0], left_arr.shape[1]))
    output = cv.medianBlur(np.float32(output), 5)
    plt.imsave("output.png", output, cmap='jet')