#include <perceptron.h>

__global__ void perceptron(float *inputs, float *weights, float *output) {
    float sum = 0;
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Calculate weighted sum
    for (int i = 0; i < NUM_INPUTS; i++) {
        sum += inputs[bid * NUM_INPUTS + i] * weights[tid * NUM_INPUTS + i];
    }
    sum += weights[tid * NUM_INPUTS + NUM_INPUTS];

    // Apply activation function
    if (sum > 0) {
        output[bid] = 1.0;
    } else {
        output[bid] = 0.0;
    }
}

void print_accuracy(float *output_h, float *expected_output_h, int num_examples) {
    float correct = 0;
    for (int i = 0; i < num_examples; i++) {
        if (output_h[i] == expected_output_h[i]) {
            correct++;
        }
    }
    float accuracy = correct / num_examples * 100;
    printf("Accuracy: %.2f%%\n", accuracy);
}


int main(int argc, char **argv) {
    float *inputs_h, *weights_h, *output_h;
    float *inputs_d, *weights_d, *output_d;
    int num_blocks, num_threads;

    // Allocate memory on host
    inputs_h = (float*)malloc(sizeof(float) * NUM_INPUTS * 4);
    weights_h = (float*)malloc(sizeof(float) * NUM_WEIGHTS * 2);
    output_h = (float*)malloc(sizeof(float) * 4);

    // Initialize inputs and weights
    inputs_h[0] = 0; inputs_h[1] = 0;
    inputs_h[2] = 0; inputs_h[3] = 1;
    inputs_h[4] = 1; inputs_h[5] = 0;
    inputs_h[6] = 1; inputs_h[7] = 1;

    weights_h[0] = 0.5; weights_h[1] = -0.5; weights_h[2] = 0.2;
    weights_h[3] = 0.9; weights_h[4] = 0.8; weights_h[5] = -0.1;

    // Allocate memory on device
    cudaMalloc((void**)&inputs_d, sizeof(float) * NUM_INPUTS * 4);
    cudaMalloc((void**)&weights_d, sizeof(float) * NUM_WEIGHTS * 2);
    cudaMalloc((void**)&output_d, sizeof(float) * 4);

    // Copy inputs and weights to device
    cudaMemcpy(inputs_d, inputs_h, sizeof(float) * NUM_INPUTS * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(weights_d, weights_h, sizeof(float) * NUM_WEIGHTS * 2, cudaMemcpyHostToDevice);

    // Set number of blocks and threads per block
    num_blocks = 2;
    num_threads = NUM_WEIGHTS;

    // Launch kernel
    perceptron<<<num_blocks, num_threads>>>(inputs_d, weights_d, output_d);

    // Copy output from device to host
    cudaMemcpy(output_h, output_d, sizeof(float) * 4, cudaMemcpyDeviceToHost);

    // Print output
    printf("Output: %f %f %f %f\n", output_h[0], output_h[1], output_h[2], output_h[3]);

    float expected_output_h[4] = {0, 0, 0, 1};
    print_accuracy(output_h, expected_output_h, 4);


    // Free memory on device
    cudaFree(inputs_d);
    cudaFree(weights_d);
    cudaFree(output_d);

    // Free memory on host
    free(inputs_h);
    free(weights_h);
    free(output_h);

    return 0;
}
