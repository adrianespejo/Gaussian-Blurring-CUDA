#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#include <iostream>
#include <string>
using namespace std;

struct pixel {
    int r, g, b;
};

int main(int argc, char *argv[]) {
    int width, height, comp;
    unsigned char *image = stbi_load("/home/bscuser/Documents/Gaussian-Blurring-CUDA/images/fruits.png", &width, &height, &comp, STBI_rgb);

    if (image == nullptr) {
        string message = "Failed to load texture.";
        throw(message);
    }

    unsigned char *new_image;

    for (int i = 0; i < width*height; i+=3) {
        int r = image[i] - '0';
        int g = image[i+1] - '0';
        int b = image[i+2] - '0';


    }

// escribir
    stbi_write_jpg("/home/bscuser/Documents/Gaussian-Blurring-CUDA/images/res2.jpg", width, height, STBI_rgb, image, 255);


}
