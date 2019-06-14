CUDA_HOME   = /Soft/cuda/8.0.61


NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -arch=compute_35 --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
#NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -arch=sm_20 --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -arch=compute_35 -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
#LD_FLAGS    = -arch=sm_20 -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
PROG_FLAGS  = -DPINNED=0

EXE00	    = main00.exe
EXE002	    = main00_v2.exe
EXE10	    = main10.exe
EXE102	    = main10_v2.exe
EXE103      = main10_v3.exe

OBJ00	    = main00.o
OBJ002	    = main00_v2.o
OBJ10	    = main10.o
OBJ102	    = main10_v2.o
OBJ103      = main10_v3.o

default: $(EXE00) $(EXE10) $(EXE102) $(EXE103) $(EXE002)

main00.o: main00.cu stb_image.h stb_image_write.h
	$(NVCC) -std=c++11 -Xcompiler=-std=c++11 -c -o $@ main00.cu $(NVCC_FLAGS) $(PROG_FLAGS)

main00_v2.o: main00_v2.cu stb_image.h stb_image_write.h
	$(NVCC) -std=c++11 -Xcompiler=-std=c++11 -c -o $@ main00_v2.cu $(NVCC_FLAGS) $(PROG_FLAGS)

main10.o: main10.cu stb_image.h stb_image_write.h
	$(NVCC) -std=c++11 -Xcompiler=-std=c++11 -c -o $@ main10.cu $(NVCC_FLAGS) $(PROG_FLAGS)

main10_v2.o: main10_v2.cu stb_image.h stb_image_write.h
	$(NVCC) -std=c++11 -Xcompiler=-std=c++11 -c -o $@ main10_v2.cu $(NVCC_FLAGS) $(PROG_FLAGS)

main10_v3.o: main10_v3.cu stb_image.h stb_image_write.h
	$(NVCC) -std=c++11 -Xcompiler=-std=c++11 -c -o $@ main10_v3.cu $(NVCC_FLAGS) $(PROG_FLAGS)

$(EXE00): $(OBJ00)
	$(NVCC) $(OBJ00) -o $(EXE00) $(LD_FLAGS)
	#$(NVCC) -O1 -v -lrt -lm -arch=sm_20 -o $(EXE00)

$(EXE002): $(OBJ002)
	$(NVCC) $(OBJ002) -o $(EXE002) $(LD_FLAGS)
	#$(NVCC) -O1 -v -lrt -lm -arch=sm_20 -o $(EXE002)

$(EXE10): $(OBJ10)
	$(NVCC) $(OBJ10) -o $(EXE10) $(LD_FLAGS)

$(EXE102): $(OBJ102)
	$(NVCC) $(OBJ102) -o $(EXE102) $(LD_FLAGS)

$(EXE103): $(OBJ103)
	$(NVCC) $(OBJ103) -o $(EXE103) $(LD_FLAGS)

all:	$(EXE00) $(EXE10) $(EXE102) $(EXE103) $(EXE002)

clean:
	rm -rf *.o main*.exe main.e1* main.o1*


