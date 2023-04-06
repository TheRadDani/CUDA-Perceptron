COMPILER=nvcc
IDIR=./headers/
STD:=c++17
EXE_DIR=./exe/
SOURCE_DIR=./src/
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -lcuda --std $(STD)

.PHONY: clean build run

build: $(SOURCE_DIR)perceptron.cu $(IDIR)perceptron.h
	$(COMPILER) $(COMPILER_FLAGS) $(SOURCE_DIR)perceptron.cu -o $(EXE_DIR)perceptron.exe


clean:
	rm -f $(EXE_DIR)perceptron.exe

run:
	$(EXE_DIR)perceptron.exe

all: clean build run