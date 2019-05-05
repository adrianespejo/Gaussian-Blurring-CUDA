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
//string root = "C:\\Users\\adrie\\OneDrive\\Documentos\\UNI\\TGA\\proyecto\\Gaussian-Blurring-CUDA\\images\\";


struct pixel_int_t {
    int r, g, b;
};

struct pixel_double_t {
    double r, g, b;
};

int kernel[5][5] ={{1, 4,  7,  4,  1},
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

void WRITE(const string &imageName, int width, int height, int comp, const void *data, int quality) {
    string imagePath = root + imageName + ".jpg";
    char path2[imagePath.length() + 1];
    strcpy(path2, imagePath.c_str());
    stbi_write_jpg(path2, width, height, comp, data, quality);
}

pixel_int_t** transformImage(const unsigned char* image, int width, int height){
    pixel_int_t** ret;
    ret = new pixel_int_t*[height];

    for(int i = 0; i<height; ++i){
        ret[i] = new pixel_int_t[width];
        int jj = 0;
        for(int j = 0; j<width; j++){
            jj = j*3;
            ret[i][j].r = image[i * width*3 + jj] - '0';
            ret[i][j].g = image[i * width*3 + jj+1] - '0';
            ret[i][j].b = image[i * width*3 + jj+2] - '0';
        }
    }
    return ret;
}

int main(int argc, char *argv[]) {

    int width, height, comp;
    string imageName = "fruits.png";
    unsigned char *image = LOAD(imageName, &width, &height, &comp, STBI_rgb);

    if (image == nullptr) {
        throw std::runtime_error("ERROR loading: " + root + imageName);
    }
    auto *new_image = (unsigned char *) malloc(height * width * 3 * sizeof(unsigned char));

    pixel_int_t **original = transformImage(image, width, height);
    const pixel_int_t nullPixel = {.r=0, .g=0, .b=0};


    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {

            int subindex = j * 3;
            pixel_int_t submatrix[5][5];

            int p = 0, q = 0;

            for (int ii = i - 2; ii <= i + 2; ii++) {
                for (int jj = j - 2; jj <= j + 2; jj ++) {
                    pixel_int_t pixel_b = nullPixel;
                    if (not (ii < 0 || jj < 0 || ii >= height || jj >= width)){
                       pixel_b = {
                           .r = original[ii][jj].r,
                           .g = original[ii][jj].g,
                           .b = original[ii][jj].b
                       };
                    }
                    submatrix[p][q] = pixel_b;
                    q++;
                }
                p++;
            }

            // compute submatrix * kernel
            pixel_int_t res[5][5];
            for (int m = 0; m < 5; m++) {
                for (int n = 0; n < 5; n++) {
                    res[m][n] = {.r = 0, .g = 0, .b = 0};
                    for (int k = 0; k < 5; k++) {
                        res[m][n].r += submatrix[m][k].r * kernel[k][n];
                        res[m][n].g += submatrix[m][k].g * kernel[k][n];
                        res[m][n].b += submatrix[m][k].b * kernel[k][n];
                    }
                }
            }
            pixel_double_t sum = {.r = 0.0, .g = 0.0, .b = 0.0};
            for (auto &row : res) {
                for (auto &pos : row) {
                    sum.r += (double) pos.r;
                    sum.g += (double) pos.g;
                    sum.b += (double) pos.b;
                }
            }


            new_image[i * (width*3) + subindex] = (unsigned char) (sum.r / 273.0) + '0';
            new_image[i * (width*3) + subindex + 1] = (unsigned char) (sum.g / 273.0) + '0';
            new_image[i * (width*3) + subindex + 2] = (unsigned char) (sum.b / 273.0) + '0';

        }

    }

    WRITE("result", width, height, STBI_rgb, image, 255);
    WRITE("blurred_fruits", width, height, STBI_rgb, new_image, 255);

    free(image);
    free(new_image);
}
