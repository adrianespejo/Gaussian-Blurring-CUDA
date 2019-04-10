#ifndef GAUSSIAN_BLURRING_CUDA_IMAGE_H
#define GAUSSIAN_BLURRING_CUDA_IMAGE_H

#include <iostream>
#include <fstream>
#include <vector>

typedef std::vector<std::vector<int>> MAT;

class Image {

public:

    Image(const std::string &path, int size);

    ~Image();

    void showMatrix();

private:
    int size;
    std::string path;
    MAT matrix;

    MAT readMatrix();

};

#endif //GAUSSIAN_BLURRING_CUDA_IMAGE_H
