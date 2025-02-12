import numpy as np

class SGM:
    """Semi-Global Matching algorithm for stereo matching"""
    
    def __init__(self, kernel_size, max_disparity, penalty1, penalty2, subpixel_interpolation, masking=True):
        self.kernel_size = kernel_size
        self.kernel_half = self.kernel_size // 2
        self.max_disparity = max_disparity
        self.penalty1 = penalty1
        self.penalty2 = penalty2
        self.subpixel_interpolation = subpixel_interpolation
        self.masking = masking
    
    def _get_patch(self, y, x, img, offset=0):
        """Get the patch centered at (y, x) with the given offset"""
        y_start = y-self.kernel_half
        y_end = y+self.kernel_half+1
        x_start = x-self.kernel_half-offset
        x_end = x+self.kernel_half-offset+1
        return img[y_start:y_end, x_start:x_end]
    
    def _census_transform(self, img):
        """Compute census transform of the image"""
        height, width = img.shape
        census_values = np.zeros_like(img, dtype=np.int32)
        for y in range(self.kernel_half, height - self.kernel_half):
            for x in range(self.kernel_half, width - self.kernel_half):
                patch = self._get_patch(y, x, img)
                # If value is less than center value assign 1 otherwise assign 0 
                census_pixel_array = (patch.flatten() > img[y, x]).astype(int)
                # Convert census array to an integer by using bit shift operator
                census_values[y, x] = np.int32(census_pixel_array.dot(1 << np.arange(self.kernel_size * self.kernel_size)[::-1])) 
        return census_values
    
    def _compute_costs(self, left_census_values, right_census_values):
        """Compute matching costs using Hamming distance"""
        height, width = left_census_values.shape
        cost_volume = np.zeros(shape=(height, width, self.max_disparity), dtype=np.uint32)
        census_tmp = np.zeros_like(left_census_values, dtype=np.int32)

        for d in range(self.max_disparity):
            # The right image is shifted d pixels accross
            census_tmp[:, self.kernel_half+d:width-self.kernel_half] = right_census_values[:, self.kernel_half:width-d-self.kernel_half]
            # 1 is assigned when the bits differ and 0 when they are the same
            xor = np.bitwise_xor(left_census_values, census_tmp)
            # All the 1's are summed up to give us the number of different pixels (the cost)
            distance = np.bitwise_count(xor)
            # All the costs for that disparity are added to the cost volume
            cost_volume[:, :, d] = distance
        return cost_volume
    
    def _get_path_cost(self, slice, offset, penalties, other_dim):
        """Compute the minimum cost path for a single direction"""
        minimum_cost_path = np.zeros(shape=(other_dim, self.max_disparity), dtype=np.int32)
        minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

        for pixel_index in range(offset, other_dim):
            # Get all the minimum disparities costs from the previous pixel in the path
            previous_cost = minimum_cost_path[pixel_index - 1, :]
            # Get all the disparities costs (from the cost volume) for the current pixel
            current_cost = slice[pixel_index, :]
            costs = np.repeat(previous_cost, repeats=self.max_disparity, axis=0).reshape(self.max_disparity, self.max_disparity)
            # Add penalties to the previous pixels disparities that differ from current pixels disparities
            costs = costs + penalties
            # Find minimum costs for the current pixels disparities using the previous disparities costs + penalties 
            costs = np.amin(costs, axis=0)  
            # Current pixels disparities costs + minimum previous pixel disparities costs (with penalty) - 
            # (constant term) minimum previous cost from all disparities 
            pixel_direction_costs = current_cost + costs - np.amin(previous_cost)
            minimum_cost_path[pixel_index, :] = pixel_direction_costs

        return minimum_cost_path 
    
    def _aggregate_costs(self, cost_volume):
        """Aggregate costs in all directions"""
        height, width, _ = cost_volume.shape
        p2 = np.full(shape=(self.max_disparity, self.max_disparity), fill_value=self.penalty2, dtype=np.int32)
        p1 = np.full(shape=(self.max_disparity, self.max_disparity), fill_value=self.penalty1 - self.penalty2, dtype=np.int32)
        p1 = np.tril(p1, k=1) # keep values lower than k'th diagonal
        p1 = np.triu(p1, k=-1) # keep values higher than k'th diagonal
        no_penalty = np.identity(self.max_disparity, dtype=np.int32) * -self.penalty1 # create diagonal matrix with values -p1
        penalties = p1 + p2 + no_penalty

        south_aggregation = np.zeros(shape=(height, width, self.max_disparity), dtype=np.float32)
        north_aggregation = np.copy(south_aggregation)

        for x in range(self.kernel_half, width-self.kernel_half):
            # Takes all the rows and disparities for a single column
            south = cost_volume[:, x, :]
            # Invert the rows to get the opposite direction
            north = np.flip(south, axis=0)
            south_aggregation[:, x, :] = self._get_path_cost(south, 1, penalties, height)
            north_aggregation[:, x, :] = np.flip(self._get_path_cost(north, 1, penalties, height), axis=0)

        east_aggregation = np.copy(south_aggregation)
        west_aggregation = np.copy(south_aggregation)
        for y in range(self.kernel_half, height-self.kernel_half):
            # Takes all the column and disparities for a single row
            east = cost_volume[y, :, :]
            # Invert the columns to get the opposite direction
            west = np.flip(east, axis=0)
            east_aggregation[y, :, :] = self._get_path_cost(east, 1, penalties, width)
            west_aggregation[y, :, :] = np.flip(self._get_path_cost(west, 1, penalties, width), axis=0)

        # Combine the costs from all paths into a single aggregation volume
        aggregation_volume = np.concatenate((south_aggregation[..., None], north_aggregation[..., None], east_aggregation[..., None], west_aggregation[..., None]), axis=3)
        return aggregation_volume
    
    def _apply_subpixel_offset(self, volume, disparity):
        """Apply subpixel offset to disparity map"""
        h, w, c = volume.shape
        for i in range(self.kernel_half, h-self.kernel_half):
            for j in range(self.kernel_half, w-self.kernel_half):
                d = int(disparity[i,j])
                if 0 < d < c-1:
                    denom = volume[i,j,d-1] + volume[i,j,d+1] - 2 * volume[i,j,d]
                    if denom != 0:
                        subpixel_offset = (volume[i,j,d-1] - volume[i,j,d+1]) / (2 * denom)
                        disparity[i,j] += subpixel_offset        
        return disparity
    
    def _select_disparity(self, aggregation_volume):
        """Select disparity with minimum cost"""
        # sum up costs for all directions
        volume = np.sum(aggregation_volume, axis=3).astype(float)
        
        # returns the disparity index with the minimum cost associated with each h x w pixel
        disparity = np.argmin(volume, axis=2).astype(float)
        if self.subpixel_interpolation:
            disparity = self._apply_subpixel_offset(volume, disparity)
        return disparity
    
    def compute(self, left, right):
        """Compute disparity map"""
        left_census = self._census_transform(left)
        right_census = self._census_transform(right)
        cost_volume = self._compute_costs(left_census, right_census)
        aggregation_volume = self._aggregate_costs(cost_volume)
        disparity = np.float32(255*self._select_disparity(aggregation_volume) / self.max_disparity)
        if self.masking:
            h, w = disparity.shape
            disparity[:self.kernel_half,:] = 0 
            disparity[h-self.kernel_half:,:] = 0 
            disparity[:,w-self.kernel_half:] = 0 
            disparity[:,:self.max_disparity] = 0 
        return disparity