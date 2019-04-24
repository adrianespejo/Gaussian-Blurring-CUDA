#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


int main(int argc, char *argv[]) {
    int width, height, comp;
    unsigned char *image = stbi_load("/home/bscuser/Documents/Gaussian-Blurring-CUDA/images/fruits.png", &width, &height, &comp, STBI_rgb);

// escribir
    stbi_write_jpg("/home/bscuser/Documents/Gaussian-Blurring-CUDA/images/res2.jpg", width, height, STBI_rgb, image, 255);


}
