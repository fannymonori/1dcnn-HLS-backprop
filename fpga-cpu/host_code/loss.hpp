#include "types.hpp"
#include <type_traits>
#include <iostream>
#include <vector>
#include <cmath>

/*
* Computes the softmax function. Right now indexing is off, and it only works if N<wide_length and M=1. Needs to be fixed!
*/
void compute_softmax(std::vector<wide_type>& matrix_in, std::vector<HLSNN_DataType>& labels, std::vector<wide_type>& matrix_out, int M, int N, int wide_length) {
    matrix_out = matrix_in;
    
    float max_val = -std::numeric_limits<float>::infinity();
    std::vector<float> exp_vector(N);
    float expsum = 0.0f;

    for (int i = 0; i < M; i++) {
        //numerical stability
        /*for (int j = 0; j < N; j++) {
            int tile_index = j / wide_length;
            int tile_element = j % wide_length;
            
            max_val = std::max(max_val, float(matrix_in[i][tile_index * wide_length + tile_element]));
        }*/

        for (int j = 0; j < N; j++) {
            int tile_index = j / wide_length;
            int tile_element = j % wide_length;

            float val = float(matrix_in[i][tile_index * wide_length + tile_element]);

            //float exp_value = std::exp(val - max_val);
            float exp_value = std::exp(val);
            exp_vector[j] = exp_value;
            expsum += exp_value;
        }

        for (int j = 0; j < N; j++) {
            int tile_index = j / wide_length;
            int tile_element = j % wide_length;

            matrix_out[i][tile_index * wide_length + tile_element] = HLSNN_DataType((exp_vector[j] / expsum) - float(labels[j]));
        }
    }
}
