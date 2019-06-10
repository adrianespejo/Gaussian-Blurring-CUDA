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

// Gaussian blur
int kernel_G[5][5] = {{1, 4,  7,  4,  1},
                      {4, 16, 26, 16, 4},
                      {7, 26, 41, 26, 7},
                      {4, 16, 26, 16, 4},
                      {1, 4,  7,  4,  1}};
int kernel_G_value = 273;

// vertical blur
int kernel[5][1] = {{1},
                    {1},
                    {1},
                    {1},
                    {1}};
int kernel_value = 5;


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

// Transformamos la imagen almacenada en un vector de char
// a una estructura de 3 matrices, una para cada canal de color
void disassembleImage(
        const unsigned char *image, unsigned int *matrixR, unsigned int *matrixG, unsigned int *matrixB, int w, int h) {
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            int pos = i * w * 3 + j * 3;
            matrixR[i * w + j] = image[pos];
            matrixG[i * w + j] = image[pos + 1];
            matrixB[i * w + j] = image[pos + 2];
        }
    }
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

    // Transformamos la imagen de entrada en 3 matrices de enteros
    unsigned int matrixR[width * height];
    unsigned int matrixG[width * height];
    unsigned int matrixB[width * height];
    disassembleImage(image, matrixR, matrixG, matrixB, width, height);

    for (int times = 0; times < blurring_times; ++times) {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {

                // Para cada pixel de la imagen calculamos la submatriz de píxeles que lo rodea
                // Y obtenemos el resultado del producto ponderado de dicha submatriz por el kernel
                // Realizamos esto para cada una de las 3 submatrices de R,G,B
                unsigned int sumR = 0;
                unsigned int sumG = 0;
                unsigned int sumB = 0;
                unsigned int ansR = 0;
                unsigned int ansG = 0;
                unsigned int ansB = 0;
                int r, c;
                int margin_x = NELEMS(kernel) / 2;
                int margin_y = NELEMS(kernel[0]) / 2;
                for (int i = -margin_x; i < (margin_x + 1); i++) {
                    for (int j = -margin_y; j < (margin_y + 1); j++) {
                        r = row + i;
                        c = col + j;
                        r = min(max(0, r), height - 1);
                        c = min(max(0, c), width - 1);
                        unsigned int pixelR = matrixR[r * width + c];
                        unsigned int pixelG = matrixG[r * width + c];
                        unsigned int pixelB = matrixB[r * width + c];
                        sumR += pixelR * kernel[i + margin_x][j + margin_y];
                        sumG += pixelG * kernel[i + margin_x][j + margin_y];
                        sumB += pixelB * kernel[i + margin_x][j + margin_y];
                    }
                }

                ansR = sumR / kernel_value;
                ansG = sumG / kernel_value;
                ansB = sumB / kernel_value;

                // Para evitar pequeños errores:
                if (ansR > 255) ansR = 255;
                if (ansG > 255) ansG = 255;
                if (ansB > 255) ansB = 255;
                if (ansR < 0) ansR = 0;
                if (ansG < 0) ansG = 0;
                if (ansB < 0) ansB = 0;

                // Una vez tenemos el valor del pixel borroso lo almacenamos en la imagen resultante
                new_image[row * (width * 3) + col * 3] = (unsigned char) ansR;
                new_image[row * (width * 3) + col * 3 + 1] = (unsigned char) ansG;
                new_image[row * (width * 3) + col * 3 + 2] = (unsigned char) ansB;

            }

        }
        // Si queremos seguir iterando necesitamos convertir la imagen resultante a un conjunto de 3 matrices
        disassembleImage(new_image, matrixR, matrixG, matrixB, width, height);
    }

    // Guardamos la imagen resultante
    string name = imageName + "_blurred_" + to_string(blurring_times) + "_times";
    WRITEPNG(name, width, height, STBI_rgb, new_image, 255);

    // Libreamos memoria
    free(image);
    free(new_image);

    return 0;
}
