#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <string.h>

using namespace std;

string root = "/home/raul/Dropbox/uni/Cuarto/TGA/PRACTICA/Gaussian-Blurring-CUDA/images/";
//string root = "/home/bscuser/Documents/Gaussian-Blurring-CUDA/images/";
//string root = "C:/Users/adrie/OneDrive/Documentos/UNI/TGA/proyecto/Gaussian-Blurring-CUDA/images/";

struct pixel_int_t {
    int r, g, b;
};

int kernel[5][5] = {{1, 4,  7,  4,  1},
                    {4, 16, 26, 16, 4},
                    {7, 26, 41, 26, 7},
                    {4, 16, 26, 16, 4},
                    {1, 4,  7,  4,  1}};


unsigned char *LOAD(const string &imageName, int *width, int *height, int *comp, int desired_channels) {
    string imagePath = root + imageName;
    char path[imagePath.length() + 1];
    strcpy(path, imagePath.c_str());
    return stbi_load(path, width, height, comp, desired_channels);
}

void WRITEJPG(const string &imageName, int width, int height, int comp, const void *data, int quality) {
    string imagePath = root + imageName + ".jpg";
    char path2[imagePath.length() + 1];
    strcpy(path2, imagePath.c_str());
    stbi_write_jpg(path2, width, height, comp, data, quality);
}

void WRITEPNG(const string &imageName, int width, int height, int comp, const void *data, int quality) {
    string imagePath = root + imageName + ".png";
    char path2[imagePath.length() + 1];
    strcpy(path2, imagePath.c_str());
    stbi_write_png(path2, width, height, comp, data, width * sizeof(char) * 3);
}

void disassembleImage(
        const unsigned char *image, unsigned int *matrixR, unsigned int *matrixG, unsigned int *matrixB, int w, int h){
    for(int i = 0; i<h; ++i){
        for(int j = 0; j<w; ++j){
            int pos = i*w*3 + j*3;
            matrixR[i*w + j] = image[pos];
            matrixG[i*w + j] = image[pos+1];
            matrixB[i*w + j] = image[pos+2];
        }
    }
}

int main(int argc, char *argv[]) {

    int blurring_times = 10;

    if (argc == 2) {
        blurring_times = atoi(argv[1]);
    }
    int width, height, comp;
    string imageName = "fruits.png";
    unsigned char *image = LOAD(imageName, &width, &height, &comp, STBI_rgb);

    if (image == nullptr) {
        throw std::runtime_error("ERROR loading: " + root + imageName);
    }
    auto *new_image = (unsigned char *) malloc(height * width * 3 * sizeof(unsigned char));

    unsigned int matrixR[width * height];
    unsigned int matrixG[width * height];
    unsigned int matrixB[width * height];

    disassembleImage(image, matrixR, matrixG, matrixB, width, height);

    for (int times = 0; times < blurring_times; ++times) {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {

                unsigned int sumR = 0;
                unsigned int sumG = 0;
                unsigned int sumB = 0;
                unsigned int ansR = 0;
                unsigned int ansG = 0;
                unsigned int ansB = 0;
                int r, c;
                for (int i = -2; i <= 2; i++) {
                    for (int j = -2; j <= 2; j++) {
                        r = row + i;
                        c = col + j;
                        r = min(max(0, r), width - 1);
                        c = min(max(0, c), height - 1);
                        unsigned int pixelR = matrixR[r*width + c];
                        unsigned int pixelG = matrixG[r*width + c];
                        unsigned int pixelB = matrixB[r*width + c];
                        sumR += pixelR * kernel[i + 2][j + 2];
                        sumG += pixelG * kernel[i + 2][j + 2];
                        sumB += pixelB * kernel[i + 2][j + 2];
                    }
                }
                ansR = sumR / 273;
                ansG = sumG / 273;
                ansB = sumB / 273;
                if (ansR > 255) ansR = 255;
                if (ansG > 255) ansG = 255;
                if (ansB > 255) ansB = 255;
                if (ansR < 0) ansR = 0;
                if (ansG < 0) ansG = 0;
                if (ansB < 0) ansB = 0;
                new_image[row * (width * 3) + col * 3] = (unsigned char) ansR;
                new_image[row * (width * 3) + col * 3 + 1] = (unsigned char) ansG;
                new_image[row * (width * 3) + col * 3 + 2] = (unsigned char) ansB;

            }

        }
        disassembleImage(new_image, matrixR, matrixG, matrixB, width, height);
    }


    WRITEPNG("result_SAME", width, height, STBI_rgb, image, 255);
    string name = "result_BLURRED_x_" + to_string(blurring_times);
    WRITEPNG(name, width, height, STBI_rgb, new_image, 255);

    free(image);
    free(new_image);
}
