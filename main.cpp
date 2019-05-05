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
string root = "C:\\Users\\adrie\\OneDrive\\Documentos\\UNI\\TGA\\proyecto\\Gaussian-Blurring-CUDA\\images\\";


struct pixel_t {
    int r, g, b;
};


int kernel[5][5] = {{1 / 273, 4 / 273,  7 / 273,  4 / 273,  1 / 273},
                    {4 / 273, 16 / 273, 26 / 273, 16 / 273, 4 / 273},
                    {7 / 273, 26 / 273, 41 / 273, 26 / 273, 7 / 273},
                    {4 / 273, 16 / 273, 26 / 273, 16 / 273, 4 / 273},
                    {1 / 273, 4 / 273,  7 / 273,  4 / 273,  1 / 273}};

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

    auto *new_image = (unsigned char *) malloc(width * height * sizeof(unsigned char));

    int npixels = 0;
    // one loop
//    for (int i = 0; i < width*height*3; i++){
//        new_image[i] = image[i];
//        npixels++;
//    }

    // two loops
    for (int i = 0; i < height * 3; i++) {
        for (int j = 0; j < width; j += 3) {
//          dos for
//            pixel_t pixel = {.r = image[i * height + j] - '0',
//                    .g = image[i * height + j + 1] - '0',
//                    .b = image[i * height + j + 2] - '0'};

            // i*height + j is the center of the 5*5 submatrix
            pixel_t submatrix[5][5];
            for (int i_b = i - 2; i_b <= i + 2; i_b++) {
                for (int j_b = j - 2; j_b <= j + 2; j_b++) {
                    pixel_t pixel_b = {.r = image[i_b * height + j_b] - '0',
                            .g = image[i_b * height + j_b + 1] - '0',
                            .b = image[i_b * height + j_b + 2] - '0'};
                    submatrix[i_b][j_b] = pixel_b;
                }
            }

            // compute submatrix * kernel
            pixel_t res[5][5];
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

            // a continuacion hay que "sumar", no acabo de entender si se suman todos los valores de res y ese
            // es el pixel i*height+j o si se suma cada posicion de res a cada pixel de la submatrix 5*5 de new_image

//            new_image[i * height + j] = pixel.r + '0';
//            new_image[i * height + j + 1] = pixel.g + '0';
//            new_image[i * height + j + 2] = pixel.b + '0';
//            npixels++;
        }
    }

    cout << "npixels: " << npixels << endl;
    WRITE("result", width, height, STBI_rgb, image, 255);
    WRITE("result_pruebas5", width, height, STBI_rgb, new_image, 255);

    free(image);
    free(new_image);
    return 0;
}
