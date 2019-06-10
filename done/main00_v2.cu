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

// Transformamos la imagen almacenada en un vector de char
// a una estructura de 3 matrices, una para cada canal de color
void disassembleImage(
        const unsigned char *image, unsigned int *matrixR, unsigned int *matrixG, unsigned int *matrixB, int w, int h){
    for(int i = 0; i<h; ++i){
        for(int j = 0; j<w; ++j){
            matrixR[i*w + j] = image[i*w*3 + j*3];
            matrixG[i*w + j] = image[i*w*3 + j*3 + 1];
            matrixB[i*w + j] = image[i*w*3 + j*3 + 2];
        }
    }
}


// kernel00
__global__ void Kernel00 (int width, int height, unsigned int *matrixR, unsigned int *matrixG, unsigned int *matrixB, unsigned char *new_image) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __const__ unsigned int kernel[5][5] = 
                   {{1, 4,  7,  4,  1},
                    {4, 16, 26, 16, 4},
                    {7, 26, 41, 26, 7},
                    {4, 16, 26, 16, 4},
                    {1, 4,  7,  4,  1}};
    int kernel_value = 273;
    
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
    int margin_x = NELEMS(kernel)/2;
    int margin_y = NELEMS(kernel[0])/2;
    for (int i = -margin_x; i < (margin_x + 1); i++) {
        for (int j = -margin_y; j < (margin_y + 1); j++) {
            r = row + i;
            c = col + j;
            r = min(max(0, r), height - 1);
            c = min(max(0, c), width - 1);
            unsigned int pixelR = matrixR[r*width + c];
            unsigned int pixelG = matrixG[r*width + c];
            unsigned int pixelB = matrixB[r*width + c];
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
    new_image[row * width * 3 + col * 3] = (unsigned char) ansR;
    new_image[row * width * 3 + col * 3 + 1] = (unsigned char) ansG;
    new_image[row * width * 3 + col * 3 + 2] = (unsigned char) ansB;   

}


int main(int argc, char **argv) {
    int width, height, comp;
    unsigned int numBytesRGB, numBytesC;

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

    numBytesRGB = sizeof(unsigned int) * width * height;
    numBytesC = sizeof(unsigned char) * height * width * 3;

    // Reservamos el espacio de memoria que ocupará la imagen resultante
    auto *new_image = (unsigned char *) malloc(numBytesC);

    if (new_image == nullptr) {
        throw std::runtime_error("Error in malloc.");
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
    unsigned int *deviceOriginalR;
    unsigned int *deviceOriginalG;
    unsigned int *deviceOriginalB;
    unsigned char *deviceNewImage;
    cudaMalloc((unsigned int **) &deviceOriginalR, numBytesRGB);
    cudaMalloc((unsigned int **) &deviceOriginalG, numBytesRGB);
    cudaMalloc((unsigned int **) &deviceOriginalB, numBytesRGB);
    cudaMalloc((unsigned char **) &deviceNewImage, numBytesC);

    cudaEventRecord(E0, 0);
    cudaEventSynchronize(E0);

    // Transformamos la imagen de entrada en 3 matrices de enteros
    unsigned int matrixR[width * height];
    unsigned int matrixG[width * height];
    unsigned int matrixB[width * height];
    disassembleImage(image, matrixR, matrixG, matrixB, width, height);    

    for (int i = 0; i < BLURRING_TIMES; i++){
		
        // Copiar datos desde el host en el device
        cudaMemcpy(deviceOriginalR, matrixR, numBytesRGB, cudaMemcpyHostToDevice);
        cudaMemcpy(deviceOriginalG, matrixG, numBytesRGB, cudaMemcpyHostToDevice);
        cudaMemcpy(deviceOriginalB, matrixB, numBytesRGB, cudaMemcpyHostToDevice);

        Kernel00 <<< dimGrid, dimBlock >>> (
                  width, height, deviceOriginalR, deviceOriginalG, deviceOriginalB, deviceNewImage);
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
        
        // Si queremos seguir iterando necesitamos convertir la imagen resultante a un conjunto de 3 matrices
        disassembleImage(new_image, matrixR, matrixG, matrixB, width, height);
    }

    // Liberar Memoria del device
    cudaFree(deviceOriginalR);
    cudaFree(deviceOriginalG);
    cudaFree(deviceOriginalB);
    cudaFree(deviceNewImage);

    cudaEventElapsedTime(&single_time, E0, E1);
    cudaEventElapsedTime(&half_time, E0, E2);
    cudaEventElapsedTime(&total_time, E0, E3);

    // Guardamos la imagen resultante
    WRITEPNG(resultName, width, height, STBI_rgb, new_image, 255);

    // Libreamos memoria
    free(image);
    free(new_image);

    printf("\nKERNEL\n");
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

