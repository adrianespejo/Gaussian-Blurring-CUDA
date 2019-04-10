#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

typedef vector<vector<int> > MAT;

MAT readMatrix(string path, int size){

    ifstream inFile;
    
    inFile.open(path);

	MAT image(size, vector<int>(size));
	
	cout << "READING FILE" << endl;	

	int pixel;
	int row = 0;
	int col = 0;
	while (inFile >> pixel) {
       
		image[row][col] = pixel;
		col++;
		if(col >= size){
			col = 0;
			row++;
		}
		
	}

	inFile.close();

	return image;

}

void showMatrix(MAT& mat){
	
	cout << "SHOWING MATRIX" << endl;

	int n = mat.size();
  
	int m = mat[0].size();
   
	for(int i = 0; i<n; ++i){
		for(int j = 0; j<m; ++j){
			cout << mat[i][j] << " ";
		}
		cout << endl;
	}

}

int main(){

	MAT gaussian = readMatrix("GaussianMatrix.png", 5);	
	
	showMatrix(gaussian);

}
