import numpy as np
import cv2 as cv

class OpenCV:

    def __init__(self,  kernel_size, max_disparity):
        self.matcher = cv.StereoBM_create(numDisparities=max_disparity, blockSize=kernel_size)
        self.matcher.setPreFilterType(cv.STEREO_BM_PREFILTER_XSOBEL)

    def compute(self, left, right):
        disparity = self.matcher.compute(left, right)
        return disparity.astype(np.float32) / 4.0