#include <vector>
#include <string>
#include "Image.h"

using namespace std;

typedef vector<vector<int>> MAT;


int main(int argc, char *argv[]) {
    string path;
    if (argc > 1) {
        path = argv[1];
    } else {
        path = "/home/bscuser/Documents/Gaussian-Blurring-CUDA/";
    }

    Image gaussian = Image(path + "GaussianMatrix.png", 5);

    gaussian.showMatrix();

}
