#include <stdlib.h>
#include <float.h>
#include <omp.h>

void block_matching(double *left, double* right, double* output, int height, int width, int max_disparity, int kernel_size, int use_subpixel) {
    int kernel_half = (int) kernel_size / 2;
    #pragma omp parallel for
    for(int i = kernel_half; i < height-kernel_half; i++){
        for(int j = max_disparity; j < width-kernel_half; j++){
            int best_offset = -1;
            double min_error = DBL_MAX;
            double errors[max_disparity];

            for (int offset = 0; offset < max_disparity; offset++){
                double error = 0.0;
                
                for(int x = -kernel_half; x <= kernel_half; x++){
                    for(int y = -kernel_half; y <= kernel_half; y++){
                        double diff = left[(i+y)*width + (j+x)] - right[(i+y)*width + (j+x-offset)];
                        error += diff * diff;
                    }
                }
                errors[offset] = error;
                if(error < min_error){
                    min_error = error;
                    best_offset = offset;
                }
            }
            double subpixel_offset = 0.0;
            if (use_subpixel) {
                if (best_offset == 0 || best_offset == max_disparity - 1) {
                    subpixel_offset = 0.0;
                }
                else {
                    double error_left = errors[best_offset - 1];
                    double error_right = errors[best_offset + 1];
                    subpixel_offset = 0.5 * (error_left - error_right) / (error_left - 2 * min_error + error_right);
                }
            }
            output[i*width + j] = best_offset + subpixel_offset;
        }
    }
}