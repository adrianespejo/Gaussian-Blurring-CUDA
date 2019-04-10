#include "Image.h"

Image::Image(const std::string &path, int size) {
    this->path = path;
    this->size = size;
    this->matrix = readMatrix();
}

Image::~Image() {
    this->matrix.clear();
}

MAT Image::readMatrix() {
    std::ifstream inFile;

    inFile.open(path);

    MAT image(size, std::vector<int>(size));

    std::cout << "READING FILE" << std::endl;

    int pixel;
    int row = 0;
    int col = 0;
    while (inFile >> pixel) {
        image[row][col] = pixel;
        col++;
        if (col >= size) {
            col = 0;
            row++;
        }

    }

    inFile.close();

    return image;
}

void Image::showMatrix() {
    std::cout << "SHOWING MATRIX" << std::endl;

    for (int i = 0; i < this->size; ++i) {
        for (int j = 0; j < this->size; ++j) {
            std::cout << this->matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}