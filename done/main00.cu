 #define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#define NELEMS(x) (sizeof(x) / sizeof((x)[0]))

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <string>

using namespace std;

#define SIZE 32

#ifndef PINNED
#define PINNED 0
#endif

// definición de un pixel con sus tres canales de color (R,G,B)
struct pixel_int_t {
    int r, g, b;
};


// funcion auxiliar para importar una imagen
unsigned char *LOAD(const string &imageName, int *width, int *height, int *comp, int desiredeviceNewImagehannels) {
    string imagePath = imageName;
    char path[imagePath.length() + 1];
    strcpy(path, imagePath.c_str());
    return stbi_load(path, width, height, comp, desiredeviceNewImagehannels);
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
void transformImage(const unsigned char *image, int width, int height, pixel_int_t *ret) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; j++) {
            int jj = j * 3;
            ret[i*width+j].r = image[i * width * 3 + jj];
            ret[i*width+j].g = image[i * width * 3 + jj + 1];
            ret[i*width+j].b = image[i * width * 3 + jj + 2];
        }
    }
}

__global__ void Kernel00 (int width, int height, pixel_int_t *original, unsigned char *new_image) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __const__ int kernel[5][5] =
                   {{1, 4,  7,  4,  1},
                    {4, 16, 26, 16, 4},
                    {7, 26, 41, 26, 7},
                    {4, 16, 26, 16, 4},
                    {1, 4,  7,  4,  1}};
    int kernel_value = 273;
        
                    
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
            pixel_int_t pixel = {.r=0, .g=0, .b=0};
            pixel = original[r * width + c];
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


int main(int argc, char **argv) {
    int width, height, comp;
    unsigned int numBytesA, numBytesC;

    unsigned int nBlocks, nThreads;

    float single_time, half_time, total_time;
    cudaEvent_t E0, E1, E2, E3;

    string imageName, resultName;
    
    // 'blurring_times' es la cantidad de iteraciones que hará el algoritmo
    // si no lo recibimos como parámetro le asignamos 10 por defecto
    int BLURRING_TIMES = 10;

    if (argc == 1) {
        imageName = "fruits.png";
        resultName = "result";
    } else if (argc == 2) {
        imageName = argv[1];
        resultName = "result";
    } else if (argc == 3) {
        imageName = argv[1];
        resultName = argv[2];
    } else if (argc == 4) {
        imageName = argv[1];
        resultName = argv[2];
        BLURRING_TIMES = atoi(argv[3]);
    } else {
        printf("Usage: ./exe IMAGENAME RESULTNAME BLURRING_TIMES\n");
        exit(0);
    }

    // Cargamos la imagen que vamos a utilizar
    unsigned char *image = LOAD(imageName, &width, &height, &comp, STBI_rgb);
    if (image == nullptr) {
        throw std::runtime_error("ERROR loading: " + imageName);
    }

    numBytesA = sizeof(pixel_int_t) * width * height;
    numBytesC = sizeof(unsigned char) * height * width * 3;

    // Reservamos el espacio de memoria que ocupará la imagen resultante
    auto *new_image = (unsigned char *) malloc(numBytesC);

    if (new_image == nullptr) {
        throw std::runtime_error("Error in malloc.\n");
    }


    // numero de Threads en cada dimension
	nThreads = SIZE;
	
    // numero de Blocks en cada dimension
    nBlocks = width/nThreads;
	
    dim3 dimGrid(nBlocks, nBlocks, 1);
    dim3 dimBlock(nThreads, nThreads, 1);

    cudaEventCreate(&E0);
    cudaEventCreate(&E1);
    cudaEventCreate(&E2);
    cudaEventCreate(&E3);

    // Obtener Memoria en el device
    pixel_int_t *deviceOriginal;
    unsigned char *deviceNewImage;
    cudaMalloc((pixel_int_t **) &deviceOriginal, numBytesA);
    cudaMalloc((unsigned char **) &deviceNewImage, numBytesC);

    cudaEventRecord(E0, 0);
    cudaEventSynchronize(E0);

    pixel_int_t *original = new pixel_int_t[width*height];
    for (int i = 0; i < BLURRING_TIMES; i++){
	    if (i > 0) {
            // Transformamos la imagen de entrada en una matriz de píxeles
            transformImage(new_image, width, height, original);
        }
        else {
            // Si queremos seguir iterando necesitamos convertir la imagen resultante a una matriz de píxeles
            transformImage(image, width, height, original);
        }

        // Copiar datos desde el host en el device
        cudaMemcpy(deviceOriginal, original, numBytesA, cudaMemcpyHostToDevice);

        // Llamada al kernel
        Kernel00 <<< dimGrid, dimBlock >>> (width, height, deviceOriginal, deviceNewImage);

        // Obtener el resultado desde el host
        cudaMemcpy(new_image, deviceNewImage, numBytesC, cudaMemcpyDeviceToHost);       
	
	    if(i+1 == 1){
	        cudaEventRecord(E1, 0);
	        cudaEventSynchronize(E1);
	    }
	    else if(i+1 == BLURRING_TIMES/2){
	        cudaEventRecord(E2, 0);
	        cudaEventSynchronize(E2);	
	    }
	    else if(i+1 == BLURRING_TIMES){
	        cudaEventRecord(E3, 0);
	        cudaEventSynchronize(E3);	
	    }	
    }

    // Liberar Memoria del device
    cudaFree(deviceOriginal);
    cudaFree(deviceNewImage);

    cudaEventElapsedTime(&single_time, E0, E1);
    cudaEventElapsedTime(&half_time, E0, E2);
    cudaEventElapsedTime(&total_time, E0, E3);

    // Guardamos la imagen resultante
    WRITEPNG(resultName, width, height, STBI_rgb, new_image, 255);

    // Libreamos memoria
    free(image);
    free(original);
    free(new_image);

    printf("\nKERNEL con imagen %dx%d\n", width, height);
    printf("nThreads: %dx%d (%d)\n", nThreads, nThreads, nThreads * nThreads);
    printf("nBlocks: %dx%d (%d)\n", nBlocks, nBlocks, nBlocks * nBlocks);
    if (PINNED) printf("Usando Pinned Memory\n");
    else printf("NO usa Pinned Memory\n");
    printf("Tiempo 1 iteracion: %4.6f milseg\n", single_time);
    printf("Tiempo %d iteraciones: %4.6f milseg\n",  BLURRING_TIMES/2, half_time);
    printf("Tiempo %d iteraciones: %4.6f milseg\n",  BLURRING_TIMES, total_time);

    cudaEventDestroy(E0);
    cudaEventDestroy(E1);
    cudaEventDestroy(E2);
    cudaEventDestroy(E3);

}
