#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <string.h>

using namespace std;

struct pixel_int_t {
    int r, g, b;
};

int kernel[5][5] = {{1, 4,  7,  4,  1},
                    {4, 16, 26, 16, 4},
  /* gauss */       {7, 26, 41, 26, 7},
                    {4, 16, 26, 16, 4},
                    {1, 4,  7,  4,  1}};


unsigned char *LOAD(const string &imageName, int *width, int *height, int *comp, int desired_channels) {
    string imagePath = root + imageName;
    char path[imagePath.length() + 1];
    strcpy(path, imagePath.c_str());
    return stbi_load(path, width, height, comp, desired_channels);
}

void WRITEPNG(const string &imageName, int width, int height, int comp, const void *data, int quality) {
    char path2[imagePath.length() + 1];
    strcpy(path2, imagePath.c_str());
    stbi_write_png(path2, width, height, comp, data, width * sizeof(char) * 3);
}

pixel_int_t *transformImage(const unsigned char *image, int width, int height) {
    pixel_int_t *ret;
    ret = new pixel_int_t [height*width];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; j++) {
            int jj = j * 3;
            ret[i*width+j].r = image[i * width * 3 + jj];
            ret[i*width+j].g = image[i * width * 3 + jj + 1];
            ret[i*width+j].b = image[i * width * 3 + jj + 2];
        }
    }
    return ret;
}

int main(int argc, char *argv[]) {

    // Default input parameters
    int blurring_times = 1;
    string imageName =  "fruits.png";
    string resultName = "result";
    
    // Loading input parameters
    if (argc == 3) {
        imageName = argv[1];
        resultName = argv[2];
    } else if (argc == 4) {
        imageName = argv[1];
        resultName = argv[2];
        blurredTimes = atoi(argv[3]);
    } else {
        printf("Usage: ./exe ORIGINAL RESULT BLURRING_TIMES (>0)\n");
        exit(0);
    }
    if (blurredTimes <= 0){
        printf("Usage: BLURRING_TIMES must be > 0\n");
        exit(0);
    }
    
    
    // Loading input image
    int width, height, comp;
    unsigned char *image = LOAD(imageName, &width, &height, &comp, STBI_rgb);
    if (image == nullptr) {
        printf("ERROR loading: " + root + imageName);
        exit(0);
    }
    
    // Memory allocation for result image
    auto *new_image = (unsigned char *) malloc(height * width * 3 * sizeof(unsigned char));

    // Transforming image to a RGB struct array
    pixel_int_t *original = transformImage(image, width, height);

    for (int times = 0; times < blurring_times; ++times) {
        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {

                pixel_int_t sumX{}, ans{};
                sumX = ans = {.r=0, .g=0, .b=0};
                int r, c;
                // Computing the kernel-size submatrix from original image
                for (int i = -2; i <= 2; ++i) {
                    for (int j = -2; j <= 2; ++j) {
                        r = row + i;
                        c = col + j;
                        r = min(max(0, r), height - 1);
                        c = min(max(0, c), width - 1);
                        pixel_int_t pixel{};
                        // Checking we don't exceed limits
                        if (not(r < 0 || c < 0 || r >= height || c >= width)) {
                            pixel = original[r*width+c];
                        }
                        // Matrix multiplication Submatrix[i][j] * kernel[j][i]
                        sumX.r += pixel.r * kernel[j + 2][i + 2];
                        sumX.g += pixel.g * kernel[j + 2][i + 2];
                        sumX.b += pixel.b * kernel[j + 2][i + 2];
                    }
                }
                // Blurred pixel = Sum of all Submatrix pixels / 273
                ans.r = abs(sumX.r) / 273;
                ans.g = abs(sumX.g) / 273;
                ans.b = abs(sumX.b) / 273;
                
                // Possible accuracy errors
                if (ans.r > 255) ans.r = 255;
                if (ans.g > 255) ans.g = 255;
                if (ans.b > 255) ans.b = 255;
                if (ans.r < 0) ans.r = 0;
                if (ans.g < 0) ans.g = 0;
                if (ans.b < 0) ans.b = 0;
                
                // Storing blurred pixel RGB values in the new image
                new_image[row * (width * 3) + col * 3] = (unsigned char) ans.r;
                new_image[row * (width * 3) + col * 3 + 1] = (unsigned char) ans.g;
                new_image[row * (width * 3) + col * 3 + 2] = (unsigned char) ans.b;
            }
        }
        // Transforming the resulting image into a RGB struct array
        original = transformImage(new_image, width, height);
    }

    // Creating the PNG file of the result image
    WRITEPNG(resultName, width, height, STBI_rgb, new_image, 255);

    free(image);
    free(new_image);
}
