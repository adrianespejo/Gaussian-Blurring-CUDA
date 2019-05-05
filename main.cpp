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


struct pixel_t {
    int r, g, b;
};

unsigned char* LOAD(const string& imageName, int *width, int *height, int *comp, int desired_channels){

    string imagePath = root + imageName;
    char path[imagePath.length() + 1];
    strcpy(path, imagePath.c_str());
    return stbi_load(path, width, height, comp, desired_channels);
}

void WRITE(const string& imageName, int width, int height, int comp, const void *data, int quality){
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

    for (int i = 0; i < width*height; i+=3) {
        pixel_t pixel = {.r = image[i] - '0', .g = image[i + 1] - '0', .b = image[i + 2] - '0'};
    }

    WRITE("result",width, height, STBI_rgb, image, 255);

}
