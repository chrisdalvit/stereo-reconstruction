#include <stdio.h>
#include <stdlib.h>

void census_transform(double* image, int height, int width, int kernel_size, double* output) {
    int half_kernel = kernel_size / 2;
        for (int y = half_kernel; y < height - half_kernel; y++) {
        for (int x = half_kernel; x < width - half_kernel; x++) {
            unsigned long long census_code = 0;
            int bit_pos = 0;
            double center_value = image[y * width + x];
            for (int ky = -half_kernel; ky <= half_kernel; ky++) {
                for (int kx = -half_kernel; kx <= half_kernel; kx++) {
                    if (kx == 0 && ky == 0){
                        continue; // Skip center pixel
                    }  
                    double neighbor_value = image[(y + ky) * width + (x + kx)];
                    if (neighbor_value > center_value) {
                        census_code |= (1ULL << bit_pos);
                    }
                    bit_pos++;
                }
            }

            output[y * width + x] = (double)census_code;
        }
    }

    // Zero out border pixels
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (y < half_kernel || y >= height - half_kernel || x < half_kernel || x >= width - half_kernel) {
                output[y * width + x] = 0;
            }
        }
    }
}


void sgm(double *left, double* right, double* output, int height, int width, int max_disparity, int kernel_size) {
    census_transform(left, height, width, kernel_size, output);
}