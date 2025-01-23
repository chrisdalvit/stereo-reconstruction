#include <stdlib.h>
#include <float.h>
#include <omp.h>

void omp_block_matching(double *left, double* right, double* output, int height, int width, int max_disparity, int kernel_size) {
    int kernel_half = (int) kernel_size / 2;
    #pragma omp parallel for
    for(int i = kernel_half; i < height-kernel_half; i++){
        for(int j = max_disparity; j < width-kernel_half; j++){
            int idx = i*width + j;
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
            output[idx] = (double) best_offset * (255.0 / max_disparity);
        }
    }
}

void block_matching(double *left, double* right, double* output, int height, int width, int max_disparity, int kernel_size) {
    int kernel_half = (int) kernel_size / 2;
    for(int i = kernel_half; i < height-kernel_half; i++){
        for(int j = max_disparity; j < width-kernel_half; j++){
            int idx = i*width + j;
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
            output[idx] = (double) best_offset * (255.0 / max_disparity);
        }
    }
}
