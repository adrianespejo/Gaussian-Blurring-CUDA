#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <string.h>

using namespace std;

//string root = "/home/raul/Dropbox/uni/Cuarto/TGA/PRACTICA/Gaussian-Blurring-CUDA/images/";
//string root = "/home/bscuser/Documents/Gaussian-Blurring-CUDA/images/";
string root = "C:/Users/adrie/OneDrive/Documentos/UNI/TGA/proyecto/Gaussian-Blurring-CUDA/images/";

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

pixel_int_t **transformImage(const unsigned char *image, int width, int height) {
    pixel_int_t **ret;
    ret = new pixel_int_t *[height];

    for (int i = 0; i < height; ++i) {
        ret[i] = new pixel_int_t[width];
        int jj = 0;
        for (int j = 0; j < width; j++) {
            jj = j * 3;
            ret[i][j].r = image[i * width * 3 + jj];
            ret[i][j].g = image[i * width * 3 + jj + 1];
            ret[i][j].b = image[i * width * 3 + jj + 2];
        }
    }
    return ret;
}

int main(int argc, char *argv[]) {

    int blurring_times = 1;
    if (argc == 2) {
        blurring_times = atoi(argv[1]);
    } else blurring_times = 10;
    int width, height, comp;
    string imageName = "fruits.png";
    unsigned char *image = LOAD(imageName, &width, &height, &comp, STBI_rgb);

    if (image == nullptr) {
        throw std::runtime_error("ERROR loading: " + root + imageName);
    }
    auto *new_image = (unsigned char *) malloc(height * width * 3 * sizeof(unsigned char));

    pixel_int_t **original = transformImage(image, width, height);

    for (int times = 0; times < blurring_times; ++times) {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {

                pixel_int_t sumX{}, ans{};
                sumX = ans = {.r=0, .g=0, .b=0};
                int r, c;
                for (int i = -2; i <= 2; i++) {
                    for (int j = -2; j <= 2; j++) {
                        r = row + i;
                        c = col + j;
                        r = min(max(0, r), width - 1);
                        c = min(max(0, c), height - 1);
                        pixel_int_t pixel = original[r][c];
                        sumX.r += pixel.r * kernel[i + 2][j + 2];
                        sumX.g += pixel.g * kernel[i + 2][j + 2];
                        sumX.b += pixel.b * kernel[i + 2][j + 2];

                    }
                }
                ans.r = abs(sumX.r) / 273;
                ans.g = abs(sumX.g) / 273;
                ans.b = abs(sumX.b) / 273;
                if (ans.r > 255) ans.r = 255;
                if (ans.g > 255) ans.g = 255;
                if (ans.b > 255) ans.b = 255;
                if (ans.r < 0) ans.r = 0;
                if (ans.g < 0) ans.g = 0;
                if (ans.b < 0) ans.b = 0;
                new_image[row * (width * 3) + col * 3] = (unsigned char) ans.r;
                new_image[row * (width * 3) + col * 3 + 1] = (unsigned char) ans.g;
                new_image[row * (width * 3) + col * 3 + 2] = (unsigned char) ans.b;

            }

        }
        original = transformImage(new_image, width, height);
    }


    WRITEPNG("result_SAME", width, height, STBI_rgb, image, 255);
    string name = "result_BLURRED_x_" + to_string(blurring_times);
    WRITEPNG(name, width, height, STBI_rgb, new_image, 255);

    free(image);
    free(new_image);
}
