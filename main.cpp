#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"


#include <stdio.h>
#include <iostream>
#include <string>

using namespace std;

//string root = "/home/raul/Dropbox/uni/Cuarto/TGA/PRACTICA/Gaussian-Blurring-CUDA/images/";
//string root = "/home/bscuser/Documents/Gaussian-Blurring-CUDA/images/";
string root = "C:\\Users\\adrie\\OneDrive\\Documentos\\UNI\\TGA\\proyecto\\Gaussian-Blurring-CUDA\\images\\";


struct pixel_int_t {
    int r, g, b;
};

struct pixel_double_t {
    double r, g, b;
};


double kernel[5][5] = {{1.0 / 273.0, 4.0 / 273.0,  7.0 / 273.0,  4.0 / 273.0,  1.0 / 273.0},
                       {4.0 / 273.0, 16.0 / 273.0, 26.0 / 273.0, 16.0 / 273.0, 4.0 / 273.0},
                       {7.0 / 273.0, 26.0 / 273.0, 41.0 / 273.0, 26.0 / 273.0, 7.0 / 273.0},
                       {4.0 / 273.0, 16.0 / 273.0, 26.0 / 273.0, 16.0 / 273.0, 4.0 / 273.0},
                       {1.0 / 273.0, 4.0 / 273.0,  7.0 / 273.0,  4.0 / 273.0,  1.0 / 273.0}};

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

int main(int argc, char *argv[]) {

    int width, height, comp;
    string imageName = "fruits.png";
    unsigned char *image = LOAD(imageName, &width, &height, &comp, STBI_rgb);

    if (image == nullptr) {
        throw std::runtime_error("ERROR loading: " + root + imageName);
    }

    auto *new_image = (unsigned char *) malloc(width * height * 3 * sizeof(unsigned char));

    int npixels, npixelsi, npixelsj;
    npixels = npixelsi = npixelsj = 0;
    // one loop
//    for (int i = 0; i < width*height*3; i++){
//        new_image[i] = image[i];
//        npixels++;
//    }
    width = width * 3;
    // two loops
    for (int i = 0; i < height; i++) {
        npixelsi++;
        npixelsj = 0;
        for (int j = 0; j < width; j += 3) {
            npixelsj++;
            npixels++;
//          dos for
            // i*height + j is the center of the 5*5 submatrix
            pixel_int_t submatrix[5][5];
            int p, q;
            p = q = 0;
            for (int i_b = i - 2; i_b <= i + 2; i_b++) {
                for (int j_b = j - 2; j_b <= j + 2; j_b += 3) {
                    pixel_int_t pixel_b;
                    if (i_b < 0 || j_b < 0 || i_b >= height || j_b >= width) {
                        pixel_b = {.r = 0, .g = 0, .b = 0};
                    } else {
                        pixel_b = {.r = (int) image[i_b * width + j_b],
                                .g = (int) image[i_b * width + j_b + 1],
                                .b = (int) image[i_b * width + j_b + 2]};
                    }
                    submatrix[p][q] = pixel_b;
                    q++;
                }
                p++;
            }

            // compute submatrix * kernel
            pixel_double_t res[5][5];
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
            pixel_double_t sum = {.r = 0, .g = 0, .b = 0};
            for (auto &row : res) {
                for (auto &pos : row) {
                    sum.r += pos.r;
                    sum.g += pos.g;
                    sum.b += pos.b;
                }
            }

            new_image[i * width + j] = (char) round(sum.r);
            new_image[i * width + j + 1] = (char) round(sum.g);
            new_image[i * width + j + 2] = (char) round(sum.b);

//            pixel_int_t pixel = {.r = (int) image[i * width + j],
//                    .g = (int) image[i * width + j + 1],
//                    .b = (int) image[i * width + j + 2]};
//
//            new_image[i * width + j] = (char) pixel.r;
//            new_image[i * width + j + 1] = (char) pixel.g;
//            new_image[i * width + j + 2] = (char) pixel.b;
        }
        cout << "npixelsj: " << npixelsj << endl;
    }

    cout << "npixels: " << npixels << endl;
    cout << "npixelsi: " << npixelsi << endl;
    WRITE("result", width / 3, height, STBI_rgb, image, 255);
    WRITE("blurred_fruits15", width / 3, height, STBI_rgb, new_image, 255);

    free(image);
    free(new_image);
}
