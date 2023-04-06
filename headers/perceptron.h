#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_INPUTS 2
#define NUM_WEIGHTS (NUM_INPUTS + 1)


__global__ void perceptron(float *inputs, float *weights, float *output);
void print_accuracy(float *output_h, float *expected_output_h, int num_examples);
