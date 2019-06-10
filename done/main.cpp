#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#define NELEMS(x) (sizeof(x) / sizeof((x)[0]))

#include <stdio.h>
#include <iostream>
#include <string>
#include <string.h>

using namespace std;

// definición de un pixel con sus tres canales de color (R,G,B)
struct pixel_int_t {
    int r, g, b;
};


// Gaussian blur
int kernel_G[5][5] = {{1, 4,  7,  4,  1},
                      {4, 16, 26, 16, 4},
                      {7, 26, 41, 26, 7},
                      {4, 16, 26, 16, 4},
                      {1, 4,  7,  4,  1}};
int kernel_G_value = 273;

// horizontal blur
int kernel[1][5] =  {1, 1, 1, 1, 1};
int kernel_value = 5;

// vertical blur
int kernel_vertical[5][1] = {{1},{1},{1},{1},{1}};
int kernel_vertical_alue = 5;

// custom blur
int kernel_custom[5][5] = {{1, 1, 1, 1, 1},
                           {1, 0, 0, 0, 1},
                           {1, 0, 1, 0, 1},
                           {1, 0, 0, 0, 1},
                           {1, 1, 1, 1, 1}};
int kernel_custom_value = 17;



// funcion auxiliar para importar una imagen
unsigned char *LOAD(const string &imageName, int *width, int *height, int *comp, int desired_channels) {
    string imagePath = imageName;
    char path[imagePath.length() + 1];
    strcpy(path, imagePath.c_str());
    return stbi_load(path, width, height, comp, desired_channels);
}

// funcion auxiliar para guardar una imagen (png)
void WRITEPNG(const string &imageName, int width, int height, int comp, const void *data, int quality) {
    string imagePath = imageName + ".png";
    char path2[imagePath.length() + 1];
    strcpy(path2, imagePath.c_str());
    stbi_write_png(path2, width, height, comp, data, width * sizeof(char) * 3);
}

// Transforma una imagen almacenada en un vector de char
// a una almacenada en una matriz de pixels
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

    // 'blurring_times' es la cantidad de iteraciones que hará el algoritmo
    // si no lo recibimos como parámetro le asignamos 10 por defecto
    int blurring_times = 10;
    string imageName = "/images/fruits.png";

    if (argc == 3) {
        blurring_times = atoi(argv[1]);
        imageName = argv[2];
    }

    int width, height, comp;

    // Cargamos la imagen que vamos a utilizar
    unsigned char *image = LOAD(imageName, &width, &height, &comp, STBI_rgb);

    if (image == nullptr) {
        throw std::runtime_error("ERROR loading: " + imageName);
    }

    // Reservamos el espacio de memoria que ocupará la imagen resultante
    auto *new_image = (unsigned char *) malloc(height * width * 3 * sizeof(unsigned char));

    // Transformamos la imagen de entrada en una matriz de píxeles
    pixel_int_t **original = transformImage(image, width, height);

    for (int times = 0; times < blurring_times; ++times) {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {

                // Para cada pixel de la imagen calculamos la submatriz de píxeles que lo rodea
                // Y obtenemos el resultado del producto ponderado de dicha submatriz por el kernel
                pixel_int_t sumX{}, ans{};
                sumX = ans = {.r=0, .g=0, .b=0};
                int r, c;
                int margin_x = NELEMS(kernel)/2;
                int margin_y = NELEMS(kernel[0])/2;
                for (int i = -margin_x; i < (margin_x + 1); i++) {
                    for (int j = -margin_y; j < (margin_y + 1); j++) {
                        r = row + i;
                        c = col + j;
                        r = min(max(0, r), height - 1);
                        c = min(max(0, c), width - 1);
                        pixel_int_t pixel{};
                        if (not(r < 0 || c < 0 || r >= height || c >= width)) pixel = original[r][c];
                        sumX.r += pixel.r * kernel[i + margin_x][j + margin_y];
                        sumX.g += pixel.g * kernel[i + margin_x][j + margin_y];
                        sumX.b += pixel.b * kernel[i + margin_x][j + margin_y];

                    }
                }
                ans.r = abs(sumX.r) / kernel_value;
                ans.g = abs(sumX.g) / kernel_value;
                ans.b = abs(sumX.b) / kernel_value;

                // Para evitar pequeños errores:
                if (ans.r > 255) ans.r = 255;
                if (ans.g > 255) ans.g = 255;
                if (ans.b > 255) ans.b = 255;
                if (ans.r < 0) ans.r = 0;
                if (ans.g < 0) ans.g = 0;
                if (ans.b < 0) ans.b = 0;

                // Una vez tenemos el valor del pixel borroso lo almacenamos en la imagen resultante
                new_image[row * (width * 3) + col * 3] = (unsigned char) ans.r;
                new_image[row * (width * 3) + col * 3 + 1] = (unsigned char) ans.g;
                new_image[row * (width * 3) + col * 3 + 2] = (unsigned char) ans.b;

            }

        }
        // Si queremos seguir iterando necesitamos convertir la imagen resultante a una matriz de píxeles
        original = transformImage(new_image, width, height);
    }


    // Guardamos la imagen resultante
    string name = imageName + "_blurred_" + to_string(blurring_times) + "_times";
    WRITEPNG(name, width, height, STBI_rgb, new_image, 255);

    // Libreamos memoria
    free(image);
    free(new_image);

    return 0;
}
