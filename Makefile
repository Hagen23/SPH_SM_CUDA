CFLAGS = -std=c++11 -Xcompiler -ansi -Wno-deprecated-gpu-targets
LDFLAGS = -lGL -lglut -lGLU
OUTF = build/
ARCH = -arch=sm_20
MKDIR_P = mkdir -p
INC = -I /usr/local/cuda/include -I cuda_common/ -I Math3D/

objects = Solver.o Solver_gpu.o Math3D/m3Matrix.o Math3D/m9Matrix.o main.o

all: $(objects) $(OUTF) 
		nvcc $(CFLAGS) $(ARCH) $(objects) -o $(OUTF)sph $(LDFLAGS)

$(OUTF):
	$(MKDIR_P) $(OUTF)

%.o: %.cu
		nvcc $(CFLAGS) $(ARCH) $(INC) -dc $< -o $@

%.o: %.cpp
		nvcc $(CFLAGS) $(ARCH) $(INC) -dc $< -o $@

clean:
		rm -f *.o $(OUTF)sph Math3D/*.o