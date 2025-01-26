import ctypes
import pathlib
import numpy as np

class BM:
    
    def __init__(self, kernel_size, max_disparity, subpixel_interpolation, language):
        self.kernel_size = kernel_size
        self.max_disparity = max_disparity
        self.kernel_half = self.kernel_size // 2
        self.offset_adjust = 255 / self.max_disparity  # this is used to map depth map output to 0-255 range
        self.subpixel_interpolation = subpixel_interpolation
        self.language = language
        
    def _get_window(self, y, x, img, offset=0):
        y_start = y-self.kernel_half
        y_end = y+self.kernel_half
        x_start = x-self.kernel_half-offset+1
        x_end = x+self.kernel_half-offset+1
        return img[y_start:y_end,x_start:x_end]
    
    def _compute_subpixel_offset(self, best_offset, errors):
        # Check if best_offset is not on border and if neighbors exist
        if 0 < best_offset < self.max_disparity-1 and errors[best_offset-1] and errors[best_offset+1]:
            denom = errors[best_offset-1] + errors[best_offset+1] - 2*errors[best_offset]
            if denom != 0:
                subpixel_offset = (errors[best_offset-1] - errors[best_offset+1]) / (2*denom)
                return subpixel_offset
        return 0.0
    
    def compute_python(self, left, right):
        h, w = left.shape
        disp_map = np.zeros_like(left, dtype=np.float32)
        for y in range(self.kernel_half, h - self.kernel_half):      
            for x in range(self.max_disparity, w - self.kernel_half):
                best_offset = None
                min_error = float("inf")
                errors = []
                for offset in range(self.max_disparity):               
                    W_left = self._get_window(y, x, left)
                    W_right = self._get_window(y, x, right, offset)
                    if W_left.shape != W_right.shape:
                        errors.append(None)
                        continue
                    error = np.sum((W_left - W_right)**2)
                    errors.append(np.float32(error))
                    if error < min_error:
                        min_error = error
                        best_offset = offset
                if self.subpixel_interpolation:
                    best_offset += self._compute_subpixel_offset(best_offset, errors)
                disp_map[y, x] = best_offset * self.offset_adjust
        return disp_map
    
    def compute_c(self, left, right):
        left = np.double(left)
        right = np.double(right)
        output = np.double(np.ones_like(left))
        # Load the shared library into ctypes
        libname = pathlib.Path().absolute() / "c/block_matching.so"
        c_lib = ctypes.CDLL(libname)
        # Define the argument and return types for the function
        c_lib.block_matching.argtypes = [
            ctypes.POINTER(ctypes.c_double), # left
            ctypes.POINTER(ctypes.c_double), # right
            ctypes.POINTER(ctypes.c_double), # output
            ctypes.c_int, # height
            ctypes.c_int, # width
            ctypes.c_int, # max disparity
            ctypes.c_int, # kernel_size
            ctypes.c_int # use_subpixel
        ]
        c_lib.block_matching.restype = ctypes.c_void_p
        left_ctypes = left.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        right_ctypes = right.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        out_ctypes = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        height = ctypes.c_int(left.shape[0])
        width = ctypes.c_int(left.shape[1])
        max_disparity = ctypes.c_int(self.max_disparity)
        kernel_size = ctypes.c_int(self.kernel_size)
        use_subpixel = ctypes.c_int(int(self.subpixel_interpolation))
        c_lib.block_matching(left_ctypes, right_ctypes, out_ctypes, height, width, max_disparity, kernel_size, use_subpixel)
        output = np.ctypeslib.as_array(out_ctypes, shape=(left.shape[0], left.shape[1]))
        return np.float32(output)

    def compute(self, left, right):
        if self.language == "python":
            return self.compute_python(left, right)
        elif self.language == "c":
            return self.compute_c(left, right)
