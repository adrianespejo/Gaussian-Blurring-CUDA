// from MM01.cu

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>

#define SIZE 32

#ifndef PINNED
#define PINNED 0
#endif

struct pixel_int_t {
    int r, g, b;
};

struct pixel_double_t {
    double r, g, b;
};

int kernel[5][5] = {{1, 4,  7,  4,  1},
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

pixel_int_t **transformImage(const unsigned char *image, int width, int height) {
    pixel_int_t **ret;
    ret = new pixel_int_t *[height];

    for (int i = 0; i < height; ++i) {
        ret[i] = new pixel_int_t[width];
        int jj = 0;
        for (int j = 0; j < width; j++) {
            jj = j * 3;
            ret[i][j].r = image[i * width * 3 + jj] - '0';
            ret[i][j].g = image[i * width * 3 + jj + 1] - '0';
            ret[i][j].b = image[i * width * 3 + jj + 2] - '0';
        }
    }
    return ret;
}

// Matriz por Matriz
// C(NxM) <- A(NxP) * B (PxM)
// (pixel_int_t **) A;
// (int **) B
// (unsigned char **) C
__global__ void Kernel10(int width, int height, pixel_int_t *A, int *B, unsigned char *C) {

    __shared__ float sA[SIZE][SIZE];
    __shared__ float sB[SIZE][SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * SIZE + ty;
    int col = bx * SIZE + tx;



    float tmp = 0.0;
    for (int m = 0; m < P; m = m + SIZE) {
        sA[ty][tx] = A[row * P + m + tx];
        sB[ty][tx] = B[col + (m + ty) * M];
        __syncthreads();

        for (int k = 0; k < SIZE; k++)
            tmp += sA[ty][k] * sB[k][tx];

        __syncthreads();
    }
    C[row * M + col] = tmp;
}


// Invocacion:
// ./ejecutable N M P test
// TAM es el la dimension de las matrices
// test == 'Y', comprueba que el resultado sea correcto
// test == 'N', NO comprueba que el resultado (Util para tomar tiempos)
// Por defecto, N = 639, M = 641, P = 1023, test == 'N'

int main(int argc, char **argv) {
    int width, height, comp, blurredTimes;
    unsigned int numBytesA, numBytesB, numBytesC;
    unsigned int nBlocksN, nBlocksM, nThreads;

    float TiempoTotal, TiempoKernel;
    cudaEvent_t E0, E1, E2, E3;

    string imageName, resultName;

    if (argc == 1) {
        imageName = "fruits.png";
        resultName = "result";
        blurredTimes = 1;
    } else if (argc == 2) {
        imageName = argv[1];
        resultName = "result";
        blurredTimes = 1;
    } else if (argc == 3) {
        imageName = argv[1];
        resultName = argv[2];
        blurredTimes = 1;
    } else if (argc == 4) {
        imageName = argv[1];
        resultName = argv[2];
        blurredTimes = atoi(argv[3]);
    } else {
        printf("Usage: ./exe IMAGENAME RESULTNAME\n");
        exit(0);
    }

    unsigned char *image = LOAD(imageName, &width, &height, &comp, STBI_rgb);
    if (image == nullptr) {
        throw std::runtime_error("ERROR loading: " + root + imageName);
    }
    pixel_int_t **original = transformImage(image, width, height);

    numBytesA = sizeof(pixel_int_t) * width * height;
    numBytesB = sizeof(int) * 5 * 5;
    numBytesC = sizeof(unsigned char) * height * width * 3;

    auto *new_image = (unsigned char *) malloc(numBytesC);

    if (new_image == nullptr) {
        throw std::runtime_error("Error in malloc.")
    }

    if (PINNED) {
        // Obtiene Memoria [pinned] en el host
        cudaMallocHost((float **) &h_A, numBytesA);
        cudaMallocHost((float **) &h_B, numBytesB);
        cudaMallocHost((float **) &h_C, numBytesC);
    } else {
        // Obtener Memoria en el host
//        h_A = (float *) malloc(numBytesA);
//        h_B = (float *) malloc(numBytesB);
//        h_C = (float *) malloc(numBytesC);
    }

    // numero de Threads en cada dimension
    nThreads = SIZE;

    // numero de Blocks en cada dimension
    nBlocksN = (width + nThreads - 1) / nThreads;
    nBlocksM = (height + nThreads - 1) / nThreads;

    dim3 dimGrid(nBlocksM, nBlocksN, 1);
    dim3 dimBlock(nThreads, nThreads, 1);

    cudaEventCreate(&E0);
    cudaEventCreate(&E1);
    cudaEventCreate(&E2);
    cudaEventCreate(&E3);

    cudaEventRecord(E0, 0);
    cudaEventSynchronize(E0);

    // Obtener Memoria en el device
    pixel_int_t *d_A;
    int *d_B;
    unsigned char *d_C;
    cudaMalloc((pixel_int_t **) &d_A, numBytesA);
    cudaMalloc((int **) &d_B, numBytesB);
    cudaMalloc((unsigned char **) &d_C, numBytesC);

    // Copiar datos desde el host en el device
    cudaMemcpy(d_A, original, numBytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, kernel, numBytesB, cudaMemcpyHostToDevice);

    cudaEventRecord(E1, 0);
    cudaEventSynchronize(E1);

    Kernel10 << < dimGrid, dimBlock >> > (width, height, d_A, d_B, d_C); // personalizar

    cudaEventRecord(E2, 0);
    cudaEventSynchronize(E2);

    // Obtener el resultado desde el host
    cudaMemcpy(new_image, d_C, numBytesC, cudaMemcpyDeviceToHost);

    // Liberar Memoria del device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventRecord(E3, 0);
    cudaEventSynchronize(E3);

    cudaEventElapsedTime(&TiempoTotal, E0, E3);
    cudaEventElapsedTime(&TiempoKernel, E1, E2);

    WRITE(resultName, width, height, STBI_rgb, new_image, 255);

    free(image);
    free(original);
    free(new_image);

    printf("\nKERNEL\n");
//    printf("Dimensiones: %dx%d <- %dx%d * %dx%d\n", N, M, N, P, P, M);
    printf("nThreads: %dx%d (%d)\n", nThreads, nThreads, nThreads * nThreads);
    printf("nBlocks: %dx%d (%d)\n", nBlocksM, nBlocksN, nBlocksN * nBlocksM);
    if (PINNED) printf("Usando Pinned Memory\n");
    else printf("NO usa Pinned Memory\n");
    printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);
    printf("Tiempo Kernel: %4.6f milseg\n", TiempoKernel);
//    printf("Rendimiento Global: %4.2f GFLOPS\n", (2.0 * (float) N * (float) M * (float) P) / (1000000.0 * TiempoTotal));
//    printf("Rendimiento Kernel: %4.2f GFLOPS\n",
//           (2.0 * (float) N * (float) M * (float) P) / (1000000.0 * TiempoKernel));

    cudaEventDestroy(E0);
    cudaEventDestroy(E1);
    cudaEventDestroy(E2);
    cudaEventDestroy(E3);

    if (PINNED) {
        cudaFreeHost(h_A);
        cudaFreeHost(h_B);
        cudaFreeHost(h_C);
    } else {
        free(h_A);
        free(h_B);
        free(h_C);
    }
}