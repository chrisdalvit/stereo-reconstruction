import numpy as np

class BM:
    
    def __init__(self, kernel_size, max_disparity, subpixel_interpolation, masking=True):
        self.kernel_size = kernel_size
        self.max_disparity = max_disparity
        self.kernel_half = self.kernel_size // 2
        self.offset_adjust = 255 / self.max_disparity  # this is used to map depth map output to 0-255 range
        self.masking = masking
        self.subpixel_interpolation = subpixel_interpolation
        
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
    
    def compute(self, left, right):
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