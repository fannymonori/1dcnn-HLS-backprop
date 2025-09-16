#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <random>
#include <fstream>
#include <algorithm>

#include "ap_fixed.h"
#include "hls_burst_maxi.h"
#include "hls_math.h"

#include "types.hpp"
#include "utils.hpp"
#include "loss.hpp"
#include "cnpy.h"

extern "C"{
    void top_mm_im2col(
            hls::burst_maxi<wide_type> out_grad,
            hls::burst_maxi<wide_type> weight,
            hls::burst_maxi<wide_type> input_grad,
            HLSNN_DataType* bias,
			hls::burst_maxi<unsigned> indices,
            unsigned mode,
            unsigned int B, unsigned int C, unsigned int F, unsigned int F_non_padded, unsigned int K,
			unsigned int col_out,
            bool do_im2col, bool do_col2im,
            bool do_relu, bool do_bias_mp,
			bool do_bias_end, bool do_maxpool, bool do_fc
    );
}

/**
Transposing matrix that is in widenes format.
*/
void transpose_matrix_widened(std::vector<wide_type> &in, std::vector<wide_type> &out, unsigned M, unsigned N, unsigned wide_length){

	unsigned M_widened = M / wide_length < 1 ? 1 : M / wide_length;
	unsigned N_widened = N / wide_length < 1 ? 1 : N / wide_length;

	unsigned index = 0;
	for(int i = 0; i < M; i++){
		unsigned col_index = 0;
		for(int j = 0; j < N_widened; j++){
			for(int k = 0; k < wide_length; k++){
				unsigned row = col_index * M_widened ;
				out[(i / wide_length) + (col_index * M_widened)][i % wide_length] = in[index][k];
				col_index++;
			}
			index++;
		}
	}
}

/**
Print matrix in a flattened format.
*/
void print_flattened_matrix(std::vector<wide_type>& flattened, int M) {
    int count = 0;
    for (size_t i = 0; i < flattened.size() && count < M; ++i) {
        for (size_t j = 0; j < WIDE_LEN && count < M; ++j) {
            std::cout << flattened[i][j] << " ";
            count++;
        }
    }
    std::cout << std::endl;
}

/**
Print matrix in a flattened format to file.
*/
void print_flattened_matrix(std::vector<wide_type>& flattened, int M, std::ofstream &out_file) {
    int count = 0;
    for (size_t i = 0; i < flattened.size() && count < M; ++i) {
        for (size_t j = 0; j < WIDE_LEN && count < M; ++j) {
            out_file << flattened[i][j] << " ";
            count++;
        }
    }
    out_file << std::endl;
}

/*
* Computes the softmax function. Right now it only works properly if N<wide_length and M=1. Needs to be fixed!
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

/**
Backward pass for ReLU.
*/
void relu_bw(std::vector<wide_type>& gradient, std::vector<wide_type>& activation, int M, int N, int wide_length) {

	unsigned loop_end = N / wide_length < 1 ? 1 : N / wide_length;
	unsigned index = 0;

	for(int i = 0; i < M; i++){
		for(int j = 0; j < loop_end; j++){
			for(unsigned k = 0; k < wide_length; k++){
                if(activation[index][k] == HLSNN_DataType(0.0)){
                	gradient[index][k] = HLSNN_DataType(0.0);
                }
			}
			index++;
		}
	}
}

/**
Backward pass for max pooling. Only works for kernel size 2 and stride 2.
*/
void maxpool_bw(std::vector<wide_type>& gradient_in, std::vector<wide_type>& gradient_out, std::vector<unsigned>& indices, int M, int N, int wide_length) {

	unsigned loop_end = N / wide_length < 1 ? 1 : N / wide_length;
	unsigned N_2 = N / 2;

	unsigned index = 0;
	unsigned indices_index = 0;

	for(int i = 0; i < M; i++){
		unsigned col_index = 0;
		for(int j = 0; j < loop_end; j++){
			for(unsigned k = 0; k < wide_length; k=k+2){
					unsigned max_index = indices[indices_index];

					HLSNN_DataType grad_val = gradient_in[indices_index / N_2][indices_index % N_2];
					gradient_out[index][k + max_index] = grad_val;
					gradient_out[index][k + (1 - max_index)] = HLSNN_DataType(0.0);
					indices_index++;
			}
			index++;
		}
	}
}

/**
Backward pass for max pooling and ReLU together. Only works for kernel size 2 and stride 2.
*/
void maxpool_relu_bw(std::vector<wide_type>& gradient_in, std::vector<wide_type>& gradient_out, std::vector<wide_type>& mp_out, std::vector<unsigned>& indices, int M, int N, int wide_length) {

	unsigned loop_end = N / wide_length < 1 ? 1 : N / wide_length;
	unsigned N_2 = N / 2;

	unsigned index = 0;
	unsigned indices_index = 0;

	for(int i = 0; i < M; i++){
		unsigned col_index = 0;
		for(int j = 0; j < loop_end; j++){
			for(unsigned k = 0; k < wide_length; k=k+2){
					unsigned max_index = indices[indices_index];
					HLSNN_DataType output_val = mp_out[indices_index / wide_length][indices_index % wide_length];
					HLSNN_DataType grad_val = gradient_in[indices_index / wide_length][indices_index % wide_length];
					gradient_out[index][k + max_index] = output_val > 0 ? grad_val : HLSNN_DataType(0.0);
					gradient_out[index][k + (1 - max_index)] = HLSNN_DataType(0.0);
					indices_index++;
			}
			index++;
		}
	}
}

/**
Print matrix that is stored in embedded vectors.
*/
template<typename T>
void print_matrix(const std::vector<std::vector<T>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& val : row) {
            std::cout << std::setw(4) << val << " ";
        }
        std::cout << "\n";
    }
}

/**
Function for reversing im2col back to original layout and flatten the matrix.
*/
void reverse_im2col_and_flatten(std::vector<wide_type>& mp_im2col, HLSNN_DataType *flat_mp_out, int M, int N, int N_mp_wide, unsigned K, int wide_length) {

	unsigned N_mp = N / 2;
	unsigned diff = N_mp_wide - N_mp;
	unsigned loop_end = N_mp / wide_length < 1 ? 1 : N_mp / wide_length;

	unsigned index = 0;

	if(N_mp % wide_length != 0){
		loop_end++;
	}

	unsigned cols = loop_end * wide_length;

	//Create temporary flat buffer
	HLSNN_DataType flat[M * K * cols];

	unsigned new_index = 0;
	unsigned flat_index = 0;

	//Flatten the widened maxpool buffer to a simple vector
	for(int i = 0; i < M; i++){
		for(int k = 0; k < K; k++){
			for(int j = 0; j < loop_end; j++){
				for(unsigned jj = 0; jj < wide_length; jj++){
					flat[flat_index] = mp_im2col[index][jj];
					flat_index++;
				}
				index++;
			}
		}
	}

	//Reverse the im2col and store result
	flat_index = 0;
	unsigned flat_index_2 = 0;
	for(int i = 0; i < M; i++){
		for(int k = 0; k < K; k++){
			flat_index_2 = i * cols + k;
			for(int j = 0; j < cols; j++){
					flat_mp_out[flat_index_2] = flat[flat_index];
					flat_index++;
					flat_index_2++;
			}
		}
	}

}

void reverse_im2col_and_flatten2(std::vector<wide_type>& mp_im2col, HLSNN_DataType *flat_mp_out, int M, int N, int N_mp_wide, unsigned K, int wide_length) {

	unsigned loop_end = N / wide_length < 1 ? 1 : N / wide_length;

	unsigned index = 0;

	if(N % wide_length != 0){
		loop_end++;
	}

	unsigned cols = loop_end * wide_length;

	//Create temporary flat buffer
	HLSNN_DataType flat[M * K * cols];

	unsigned new_index = 0;
	unsigned flat_index = 0;

	//Flatten the widened maxpool buffer to a simple vector
	for(int i = 0; i < M; i++){
		for(int k = 0; k < K; k++){
			for(int j = 0; j < loop_end; j++){
				for(unsigned jj = 0; jj < wide_length; jj++){
					flat[flat_index] = mp_im2col[index][jj];
					flat_index++;
				}
				index++;
			}
		}
	}

	//Reverse the im2col and store result
	flat_index = 0;
	unsigned flat_index_2 = 0;
	for(int i = 0; i < M; i++){
		for(int k = 0; k < K; k++){
			flat_index_2 = i * cols + k;
			for(int j = 0; j < cols; j++){
					flat_mp_out[flat_index_2] = flat[flat_index];
					flat_index++;
					flat_index_2++;
			}
		}
	}

}

/**
Backward pass for max pooling and ReLU together, while also reversing the im2col function back to original layout. This is for max pooling with pool size 2 and stride 2.
*/
void maxpool_relu_bw_im2col(std::vector<wide_type>& gradient_in, std::vector<wide_type>& gradient_out, HLSNN_DataType *mp_out, std::vector<unsigned>& indices, int M, int N, unsigned K, int C, int wide_length) {

	unsigned loop_end = N / wide_length < 1 ? 1 : N / wide_length;
	unsigned N_p2 = N / 2; //number of cols after applying max pooling
	unsigned diff = C - N_p2;
	unsigned loop_end2 = N_p2 / wide_length < 1 ? 1 : N_p2 / wide_length;

	if(N_p2 % wide_length != 0){
		loop_end2++;
	}

	unsigned new_index = 0;
	unsigned indices_index = 0;
	unsigned mp_index = 0;
	unsigned flat_index = 0;
	unsigned index = 0;

	for(int i = 0; i < M; i++){
		for(int j = 0; j < loop_end; j++){
			for(unsigned jj = 0; jj < wide_length; jj=jj+2){
				unsigned max_index = indices[mp_index];
				HLSNN_DataType output_val = mp_out[flat_index];
				HLSNN_DataType grad_val = gradient_in[indices_index / wide_length][indices_index % wide_length];
				gradient_out[index][jj + max_index] = output_val > 0 ? grad_val : HLSNN_DataType(0.0);
				gradient_out[index][jj + (1 - max_index)] = HLSNN_DataType(0.0);
				indices_index++;
				mp_index++;
				flat_index++;
			}
			index++;
		}
		indices_index += diff;
		flat_index += diff;
	}
}

/**
Backward pass for ReLU together, adhering to im2col format.
*/
void relu_bw_im2col(std::vector<wide_type>& gradient_out, HLSNN_DataType* activation_in, int M, int N, unsigned K, int C, int wide_length) {
	unsigned loop_end = N / wide_length < 1 ? 1 : N / wide_length;
	unsigned flat_index = 0;
	unsigned index = 0;

	for(int i = 0; i < M; i++){
		for(int j = 0; j < loop_end; j++){
			for(unsigned jj = 0; jj < wide_length; jj++){
				HLSNN_DataType output_val = activation_in[flat_index];
				HLSNN_DataType grad_val = gradient_out[index][jj];
				gradient_out[index][jj] = output_val > 0 ? grad_val : HLSNN_DataType(0.0);
				flat_index++;
			}
			index++;
		}
	}
}

void print_flattened(HLSNN_DataType *v, unsigned rows, unsigned cols){
	std::cout << std::endl << std::endl;
	unsigned flat_index = 0;
	for(unsigned i = 0; i < rows; i++){
		for(unsigned j = 0; j < cols; j++){
			std::cout << v[flat_index] << " ";
			flat_index++;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

/**
Function for weight update with SGD of continuously stored matrix.
*/
void weight_sgd(std::vector<wide_type>& matrix, std::vector<wide_type>& gradient, int M, int N, float learning_rate, unsigned wide_length) {
    int total_elements = M * N;

    for (int i = 0; i < total_elements; i++) {
        int tile_index = i / wide_length;
        int tile_element = i % wide_length;
        matrix[tile_index][tile_element] = HLSNN_DataType(float(matrix[tile_index][tile_element]) - float(learning_rate) * float(gradient[tile_index][tile_element]));
    }
}

/**
Bias update with SGD of bias stored in vector.
*/
void bias_sgd_FC(std::vector<HLSNN_DataType>& vec, std::vector<wide_type>& gradient, unsigned M, unsigned N, unsigned N_orig, float learning_rate, unsigned wide_length) {
    HLSNN_DataType lr_ = HLSNN_DataType(learning_rate);
	unsigned loop_end = N / wide_length < 1 ? 1 : N / wide_length;
	unsigned index = 0;
	unsigned flat_index = 0;

	if(N % wide_length != 0){
		loop_end++;
	}

	for(int i = 0; i < M; i++){
		unsigned col_index = 0;
		for(int j = 0; j < loop_end; j++){
			for(unsigned jj = 0; jj < wide_length; jj++){
				if(col_index < N_orig){
					vec[flat_index] = vec[flat_index] - lr_ * gradient[index][jj];
				}
				else{
					vec[flat_index] = 0.0;
				}
				flat_index++;
				col_index++;
			}
			index++;
		}
	}
}

/**
Calculate dB for convolutional layers.
*/
void do_dB_conv(std::vector<wide_type>& gradient, std::vector<HLSNN_DataType>& b, unsigned C, unsigned C_orig, unsigned W, float learning_rate, unsigned wide_len){
    HLSNN_DataType lr_ = HLSNN_DataType(learning_rate);
    std::vector<HLSNN_DataType> b_grad;
	b_grad.resize(C, 0.0);

	unsigned loop_end = W / wide_len < 1 ? 1 : W / wide_len;
	unsigned index = 0;

	for(int i = 0; i < C; i++){
        unsigned col_index = 0;
		for(int j = 0; j < loop_end; j++){
			for(int jj = 0; jj < wide_len; jj++){
				b_grad[i] += gradient[index][jj];
                col_index++;
			}
			index++;
		}
	}

	for(int i = 0; i < C; i++){
		if(i < C_orig){
			b[i] = b[i] - lr_ * b_grad[i];
		}
		else{
			b[i] = 0.0;
		}
	}
}

/////////////////////////////////////////////////

#define CNN1D_X_LENGTH 128

//CONV1 + MP
#define CNN1D_CONV_1_C 6
#define CNN1D_CONV_1_K 2
#define CNN1D_CONV_1_F 32
#define CNN1D_CONV_1_C_im2col (6 * 2)
#define CNN1D_CONV_1_C_im2col_padded 16
#define CNN1D_CONV_1_W 128
#define CNN1D_CONV_1_MP_W 64
#define CNN1D_CONV_1_W_conv 64

//CONV2 + MP
#define CNN1D_CONV_2_C 32
#define CNN1D_CONV_2_K 2
#define CNN1D_CONV_2_F 16
#define CNN1D_CONV_2_C_im2col (32 * 2)
#define CNN1D_CONV_2_W 64
#define CNN1D_CONV_2_MP_W 32
#define CNN1D_CONV_2_W_conv 32

//CONV3
#define CNN1D_CONV_3_C 16
#define CNN1D_CONV_3_K 2
#define CNN1D_CONV_3_F 16
#define CNN1D_CONV_3_C_im2col (16 * 2)
#define CNN1D_CONV_3_W 32
#define CNN1D_CONV_3_W_conv 32

//CONV4
#define CNN1D_CONV_4_C 16
#define CNN1D_CONV_4_K 2
#define CNN1D_CONV_4_F 16
#define CNN1D_CONV_4_C_im2col (16 * 2)
#define CNN1D_CONV_4_W 32
#define CNN1D_CONV_4_W_conv 32

//CONV5
#define CNN1D_CONV_5_C 16
#define CNN1D_CONV_5_K 2
#define CNN1D_CONV_5_F 16
#define CNN1D_CONV_5_C_im2col (16 * 2)
#define CNN1D_CONV_5_W 32
#define CNN1D_CONV_5_W_conv 32

#define CNN1D_FC_1_C 512
#define CNN1D_FC_1_F 3
#define CNN1D_FC_1_F_widened 16

#define MAX_ARRAY_SIZE 3000


void test_1dcnn_pipeline(){

    std::ofstream out_file;
    out_file.open ("conv_output.txt");

    out_file << "RUN TEST FOR 1D-CNN PIPELINE" << std::endl;

    char *cwd = get_current_dir_name();
    std::string working_dir(cwd);
    free(cwd);

	std::string weights_path = working_dir + "/im2col_1dcnn_test_07_23.npz";
	//std::vector<std::string> layer_names = {"x", "y", "x_orig", "w1", "w1_tr", "w2", "w2_tr", "w3", "w3_tr", "w4", "w4_tr", "w5", "w5_tr", "b1", "b2", "b3", "b4", "b5", "fcw1", "fcw1_padded", "fcb1"};
	std::vector<std::string> layer_names = {"x", "y", "x_orig", "w1", "w2", "w3", "w4", "w5", "b1", "b2", "b3", "b4", "b5", "fcw1", "fcw1_padded", "fcb1"};
	std::map<std::string, std::vector<HLSNN_DataType>> dataMap;
	std::map<std::string, std::vector<double>> dataMap_float;

    readNpz(weights_path, layer_names, dataMap_float);

    std::map<std::string, std::vector<double>>::iterator it;
    for (it = dataMap_float.begin(); it != dataMap_float.end(); it++)
    {
    	std::vector<double> tmp = it->second;
    	std::vector<HLSNN_DataType> tmp_result;

    	std::cout << tmp.size() << std::endl;

        for(int i = 0; i < tmp.size(); i++){
        	if(std::isnan(tmp[i])){
        		std::cout << "nan" << std::endl;
        		tmp_result.push_back(HLSNN_DataType(0.0));
        	}
        	else{
        		tmp_result.push_back(HLSNN_DataType(tmp[i]));
        	}
        }

        dataMap.insert({it->first, tmp_result});
    }

    // Pointers
    HLSNN_DataType* x_ptr = dataMap["x"].data();

    HLSNN_DataType* conv1_w_ptr = dataMap["w1"].data();
    HLSNN_DataType* conv2_w_ptr = dataMap["w2"].data();
    HLSNN_DataType* conv3_w_ptr = dataMap["w3"].data();
    HLSNN_DataType* conv4_w_ptr = dataMap["w4"].data();
    HLSNN_DataType* conv5_w_ptr = dataMap["w5"].data();
    HLSNN_DataType* conv1_b_ptr = dataMap["b1"].data();
    HLSNN_DataType* conv2_b_ptr = dataMap["b2"].data();
    HLSNN_DataType* conv3_b_ptr = dataMap["b3"].data();
    HLSNN_DataType* conv4_b_ptr = dataMap["b4"].data();
    HLSNN_DataType* conv5_b_ptr = dataMap["b5"].data();

    HLSNN_DataType* fcw1_w_ptr = dataMap["fcw1"].data();
    HLSNN_DataType* fcw1_w_padded_ptr = dataMap["fcw1_padded"].data();
    HLSNN_DataType* fcw1_b_ptr = dataMap["fcb1"].data();

    HLSNN_DataType* y_ptr = dataMap["y"].data();

    std::cout << "All data read in" << std::endl;

    //////////////////////////////////////////////////////////////

    //Input storage
    std::vector<wide_type> input_storage_wide;

    //Label storage
    std::vector<HLSNN_DataType> label;

    //Weight storage declarations
    std::vector<wide_type> conv1_w_wide;
    std::vector<wide_type> conv1_w_tr_wide;
    std::vector<wide_type> conv2_w_wide;
    std::vector<wide_type> conv2_w_tr_wide;
    std::vector<wide_type> conv3_w_wide;
    std::vector<wide_type> conv3_w_tr_wide;
    std::vector<wide_type> conv4_w_wide;
    std::vector<wide_type> conv4_w_tr_wide;
    std::vector<wide_type> conv5_w_wide;
    std::vector<wide_type> conv5_w_tr_wide;
    std::vector<wide_type> fc1_w_padded_wide;
    std::vector<wide_type> fc1_w_wide;

    //Bias storage declarations
    std::vector<HLSNN_DataType> conv1_b;
    std::vector<HLSNN_DataType> conv2_b;
    std::vector<HLSNN_DataType> conv3_b;
    std::vector<HLSNN_DataType> conv4_b;
    std::vector<HLSNN_DataType> conv5_b;
    std::vector<HLSNN_DataType> fc1_b;

    //CONV 1
	place_in_wide(conv1_w_ptr, conv1_w_wide, CNN1D_CONV_1_F, CNN1D_CONV_1_C_im2col_padded, WIDE_LEN); //CONV 1 weight
	place_in_vector(conv1_b, conv1_b_ptr, CNN1D_CONV_1_F); //CONV 1 bias
	conv1_w_tr_wide.resize(CNN1D_CONV_1_F * CNN1D_CONV_1_C_im2col_padded, HLSNN_DataType(0.0));
    transpose_matrix_widened(conv1_w_wide, conv1_w_tr_wide, CNN1D_CONV_1_F, CNN1D_CONV_1_C_im2col_padded, WIDE_LEN);

    //CONV2
	place_in_wide(conv2_w_ptr, conv2_w_wide, CNN1D_CONV_2_F, CNN1D_CONV_2_C_im2col, WIDE_LEN); //CONV 2 weight
    place_in_vector(conv2_b, conv2_b_ptr, CNN1D_CONV_2_F); //CONV 2 bias
    conv2_w_tr_wide.resize(CNN1D_CONV_2_F * CNN1D_CONV_2_C_im2col, HLSNN_DataType(0.0));
    transpose_matrix_widened(conv2_w_wide, conv2_w_tr_wide, CNN1D_CONV_2_F, CNN1D_CONV_2_C_im2col, WIDE_LEN);

    //CONV3
	place_in_wide(conv3_w_ptr, conv3_w_wide, CNN1D_CONV_3_F, CNN1D_CONV_3_C_im2col, WIDE_LEN); //CONV 3 weight
    place_in_vector(conv3_b, conv3_b_ptr, CNN1D_CONV_3_F); //CONV 3 bias
    conv3_w_tr_wide.resize(CNN1D_CONV_3_F * CNN1D_CONV_3_C_im2col, HLSNN_DataType(0.0));
    transpose_matrix_widened(conv3_w_wide, conv3_w_tr_wide, CNN1D_CONV_3_F, CNN1D_CONV_3_C_im2col, WIDE_LEN);

    //CONV4
	place_in_wide(conv4_w_ptr, conv4_w_wide, CNN1D_CONV_4_F, CNN1D_CONV_4_C_im2col, WIDE_LEN); //CONV 3 weight
    place_in_vector(conv4_b, conv4_b_ptr, CNN1D_CONV_4_F); //CONV 4 bias
    conv4_w_tr_wide.resize(CNN1D_CONV_4_F * CNN1D_CONV_4_C_im2col, HLSNN_DataType(0.0));
    transpose_matrix_widened(conv4_w_wide, conv4_w_tr_wide, CNN1D_CONV_4_F, CNN1D_CONV_4_C_im2col, WIDE_LEN);

    //CONV5
	place_in_wide(conv5_w_ptr, conv5_w_wide, CNN1D_CONV_5_F, CNN1D_CONV_5_C_im2col, WIDE_LEN); //CONV 3 weight
    place_in_vector(conv5_b, conv5_b_ptr, CNN1D_CONV_5_F); //CONV 5 bias
    conv5_w_tr_wide.resize(CNN1D_CONV_5_F * CNN1D_CONV_5_C_im2col, HLSNN_DataType(0.0));
    transpose_matrix_widened(conv5_w_wide, conv5_w_tr_wide, CNN1D_CONV_5_F, CNN1D_CONV_5_C_im2col, WIDE_LEN);

    //FC1
    place_in_wide_widen(fcw1_w_padded_ptr, fc1_w_padded_wide, 512, CNN1D_FC_1_F, CNN1D_FC_1_F_widened, WIDE_LEN); //FC 1 weight padded
    place_in_vector_pad(fc1_b, fcw1_b_ptr, CNN1D_FC_1_F_widened, CNN1D_FC_1_F); //FC 1 bias


    // Print data
    out_file << std::endl << "################################## INITIAL DATA ##################################" << std::endl;

	out_file << "W1 padded:" << std::endl;
	print_output_wide(conv1_w_wide.data(), CNN1D_CONV_1_F, CNN1D_CONV_1_C_im2col_padded, WIDE_LEN, out_file);

	out_file << "b1:" << std::endl;
    print_vector(conv1_b, CNN1D_CONV_1_F, out_file);

	out_file << "W2 weight:" << std::endl;
	print_output_wide(conv2_w_wide.data(), CNN1D_CONV_2_F, CNN1D_CONV_2_C_im2col, WIDE_LEN, out_file);

	out_file << "b2:" << std::endl;
	print_vector(conv2_b, CNN1D_CONV_2_F, out_file);

	out_file << "W3 weight:" << std::endl;
	print_output_wide(conv3_w_wide.data(), CNN1D_CONV_3_F, CNN1D_CONV_3_C_im2col, WIDE_LEN, out_file);

	out_file << "W3 weight transposed:" << std::endl;
	print_output_wide(conv3_w_tr_wide.data(), CNN1D_CONV_3_C_im2col, CNN1D_CONV_3_F, WIDE_LEN, out_file);

	out_file << "b3:" << std::endl;
	print_vector(conv3_b, CNN1D_CONV_3_F, out_file);

	out_file << "W4 weight:" << std::endl;
	print_output_wide(conv4_w_wide.data(), CNN1D_CONV_4_F, CNN1D_CONV_4_C_im2col, WIDE_LEN, out_file);

	out_file << "W4 weight transposed:" << std::endl;
	print_output_wide(conv4_w_tr_wide.data(), CNN1D_CONV_4_C_im2col, CNN1D_CONV_4_F, WIDE_LEN, out_file);

	out_file << "b4:" << std::endl;
	print_vector(conv4_b, CNN1D_CONV_4_F, out_file);

	out_file << "W5 weight:" << std::endl;
	print_output_wide(conv5_w_wide.data(), CNN1D_CONV_5_F, CNN1D_CONV_5_C_im2col, WIDE_LEN, out_file);

	out_file << "W5 weight transposed:" << std::endl;
	print_output_wide(conv5_w_tr_wide.data(), CNN1D_CONV_5_C_im2col, CNN1D_CONV_5_F, WIDE_LEN, out_file);

	out_file << "b5:" << std::endl;
	print_vector(conv5_b, CNN1D_CONV_5_F, out_file);

	out_file << "FC1 weight padded:" << std::endl;
	print_output_wide(fc1_w_padded_wide.data(), 512, 16, WIDE_LEN, out_file);

	out_file << "FC1 bias:" << std::endl;
	print_vector(fc1_b, CNN1D_FC_1_F_widened, out_file);

    ///////
    //Output storage declarations
    std::vector<wide_type> mp1_storage_wide;
	std::vector<wide_type> d_conv1_storage_wide;
    std::vector<wide_type> d_mp1_storage_wide;
    std::vector<wide_type> mp2_storage_wide;
    std::vector<wide_type> d_conv2_storage_wide;
    std::vector<wide_type> d_mp2_storage_wide;

    std::vector<wide_type> conv3_storage_wide;
    std::vector<wide_type> d_conv3_storage_wide;

    std::vector<wide_type> conv4_storage_wide;
    std::vector<wide_type> d_conv4_storage_wide;
    std::vector<wide_type> d_conv4_relu_storage_wide;

    std::vector<wide_type> conv5_storage_wide;
    std::vector<wide_type> d_conv5_storage_wide;

    std::vector<wide_type> dense1_storage_wide;
    std::vector<wide_type> d_dense1_storage_wide;
    std::vector<wide_type> dense2_storage_wide;
    std::vector<wide_type> d_flat_storage_wide;

    std::vector<wide_type> softmax_storage;
    std::vector<wide_type> output_gradient_storage;

    std::vector<wide_type> d_fc1_w_wide;

    std::vector<wide_type> d_conv5_w_wide;
    std::vector<wide_type> d_conv4_w_wide;
    std::vector<wide_type> d_conv3_w_wide;
    std::vector<wide_type> d_conv2_w_wide;
    std::vector<wide_type> d_conv1_w_wide;

    std::vector<unsigned> conv1_indices;
    std::vector<unsigned> conv2_indices;
    std::vector<unsigned> conv3_indices;
    std::vector<unsigned> conv4_indices;
    std::vector<unsigned> conv5_indices;
    std::vector<unsigned> dense1_indices;

    unsigned K_ = 2;
    mp1_storage_wide.resize(CNN1D_CONV_1_F * CNN1D_CONV_1_K * CNN1D_CONV_1_W, HLSNN_DataType(0.0));
    d_mp1_storage_wide.resize(CNN1D_CONV_1_F * CNN1D_CONV_1_K * CNN1D_CONV_1_W, HLSNN_DataType(0.0));
    d_conv1_storage_wide.resize(CNN1D_CONV_1_F * CNN1D_CONV_1_K * CNN1D_CONV_1_W, HLSNN_DataType(0.0));

    mp2_storage_wide.resize(CNN1D_CONV_2_F * CNN1D_CONV_2_K * CNN1D_CONV_2_W, HLSNN_DataType(0.0));
    d_mp2_storage_wide.resize(CNN1D_CONV_2_F * CNN1D_CONV_2_K * CNN1D_CONV_2_W, HLSNN_DataType(0.0));
    d_conv2_storage_wide.resize(CNN1D_CONV_2_F * CNN1D_CONV_2_K * CNN1D_CONV_2_W, HLSNN_DataType(0.0));

    conv3_storage_wide.resize(CNN1D_CONV_3_F * CNN1D_CONV_3_K * CNN1D_CONV_3_W, HLSNN_DataType(0.0));
    d_conv3_storage_wide.resize(CNN1D_CONV_3_F * CNN1D_CONV_3_K * CNN1D_CONV_3_W, HLSNN_DataType(0.0));

    conv4_storage_wide.resize(CNN1D_CONV_4_F * CNN1D_CONV_4_K * CNN1D_CONV_4_W, HLSNN_DataType(0.0));
    d_conv4_storage_wide.resize(CNN1D_CONV_4_F * CNN1D_CONV_4_K * CNN1D_CONV_4_W, HLSNN_DataType(0.0));
    d_conv4_relu_storage_wide.resize(CNN1D_CONV_4_F * CNN1D_CONV_4_K * CNN1D_CONV_4_W, HLSNN_DataType(0.0));

    conv5_storage_wide.resize(CNN1D_CONV_5_F * K_ * CNN1D_CONV_5_W, HLSNN_DataType(0.0));
    d_conv5_storage_wide.resize(CNN1D_CONV_5_F * K_ * CNN1D_CONV_5_W, HLSNN_DataType(0.0));

    dense1_storage_wide.resize(512, HLSNN_DataType(0.0));
    d_dense1_storage_wide.resize(512, HLSNN_DataType(0.0));

    d_flat_storage_wide.resize(512, HLSNN_DataType(0.0));

    softmax_storage.resize(512, HLSNN_DataType(0.0));

    d_fc1_w_wide.resize(CNN1D_FC_1_C * 3, HLSNN_DataType(0.0));

    d_conv5_w_wide.resize(CNN1D_CONV_5_F * K_ * CNN1D_CONV_5_C, HLSNN_DataType(0.0));
    d_conv4_w_wide.resize(CNN1D_CONV_4_F * K_ * CNN1D_CONV_4_C, HLSNN_DataType(0.0));
    d_conv3_w_wide.resize(CNN1D_CONV_3_F * K_ * CNN1D_CONV_3_C, HLSNN_DataType(0.0));
    d_conv2_w_wide.resize(CNN1D_CONV_2_F * K_ * CNN1D_CONV_2_C, HLSNN_DataType(0.0));
    d_conv1_w_wide.resize(CNN1D_CONV_1_F * K_ * CNN1D_CONV_1_C, HLSNN_DataType(0.0));

    conv1_indices.resize(MAX_ARRAY_SIZE, 0);
    conv2_indices.resize(MAX_ARRAY_SIZE, 0);
    conv3_indices.resize(MAX_ARRAY_SIZE, 0);
    conv4_indices.resize(MAX_ARRAY_SIZE, 0);
    conv5_indices.resize(MAX_ARRAY_SIZE, 0);
    dense1_indices.resize(MAX_ARRAY_SIZE, 0);

    out_file << std::endl << "################################## INPUTS ##################################" << std::endl;

    place_in_wide(x_ptr, input_storage_wide, CNN1D_CONV_1_C_im2col_padded, CNN1D_CONV_1_W, WIDE_LEN);
    place_in_vector(label, y_ptr, 3); // Y

    out_file << "X padded:" << std::endl;
    print_output_wide(input_storage_wide.data(), CNN1D_CONV_1_C_im2col_padded, CNN1D_CONV_1_W, WIDE_LEN, out_file);

    out_file << "Label:" << std::endl;
	print_vector(label, 3, out_file);

    out_file << std::endl << "################################## CONV FW 1 ##################################" << std::endl;

    // [CNN_CONV_1_F, CNN_CONV_1_C_im2col_padded] X [CNN_CONV_1_C_im2col_padded, CNN_CONV_1_W]
    // [16, 32] X [32, 128]
    // [rows_in, cols_in] X [cols_in, cols_out]

    top_mm_im2col(
    					mp1_storage_wide.data(),
						input_storage_wide.data(),
    					conv1_w_wide.data(),
    					conv1_b.data(),
						conv1_indices.data(),
        				0,
						CNN1D_CONV_1_F, CNN1D_CONV_1_C_im2col_padded, CNN1D_CONV_1_W, CNN1D_CONV_1_W_conv, CNN1D_CONV_1_K, CNN1D_CONV_1_MP_W / WIDE_LEN, 1, 0, 1, 1, 0, 1, 0
    );

    print_output_wide(mp1_storage_wide.data(), CNN1D_CONV_2_C_im2col, CNN1D_CONV_2_W, WIDE_LEN, out_file);
    print_vector(conv1_indices, MAX_ARRAY_SIZE, out_file);

    HLSNN_DataType *mp1_flatten = new HLSNN_DataType [CNN1D_CONV_1_F * CNN1D_CONV_1_K * CNN1D_CONV_2_W];
    reverse_im2col_and_flatten(mp1_storage_wide, mp1_flatten, CNN1D_CONV_1_F, CNN1D_CONV_1_W, CNN1D_CONV_2_W, CNN1D_CONV_1_K, WIDE_LEN);
    out_file << std::endl << "Reverse im2col: " << std::endl;
    print_output(mp1_flatten, CNN1D_CONV_1_F, CNN1D_CONV_2_W, out_file);


    out_file << std::endl << "################################## CONV FW 2 ##################################" << std::endl;

    top_mm_im2col(
    				mp2_storage_wide.data(),
					mp1_storage_wide.data(),
					conv2_w_wide.data(),
					conv2_b.data(),
					conv2_indices.data(),
    				0,
					CNN1D_CONV_2_F, CNN1D_CONV_2_C_im2col, CNN1D_CONV_2_W, CNN1D_CONV_3_W_conv, K_, 32 / WIDE_LEN, 1, 0, 1, 1, 0, 1, 0
    );

    print_output_wide(mp2_storage_wide.data(), CNN1D_CONV_2_F * K_, CNN1D_CONV_3_W, WIDE_LEN, out_file);
    print_vector(conv2_indices, MAX_ARRAY_SIZE, out_file);

    HLSNN_DataType *conv2_flatten = new HLSNN_DataType [CNN1D_CONV_2_F * 2 * CNN1D_CONV_2_W];
    reverse_im2col_and_flatten(mp2_storage_wide, conv2_flatten, CNN1D_CONV_2_F, CNN1D_CONV_2_W, CNN1D_CONV_2_W_conv - 2, 2, WIDE_LEN);
    out_file << std::endl << "Reverse im2col: " << std::endl;
    print_output(conv2_flatten, CNN1D_CONV_2_F, CNN1D_CONV_2_W, out_file);

    out_file << std::endl << "################################## CONV FW 3 ##################################" << std::endl;

    top_mm_im2col(
    				conv3_storage_wide.data(),
					mp2_storage_wide.data(),
					conv3_w_wide.data(),
					conv3_b.data(),
					conv3_indices.data(),
    				0,
					CNN1D_CONV_3_F, CNN1D_CONV_3_C_im2col, CNN1D_CONV_3_W, CNN1D_CONV_3_W_conv - 1, 2, 32 / WIDE_LEN, 1, 0, 1, 0, 0, 0, 0
    );

    print_output_wide(conv3_storage_wide.data(), CNN1D_CONV_3_F * 2, 32, WIDE_LEN, out_file);

    HLSNN_DataType *conv3_flatten = new HLSNN_DataType [CNN1D_CONV_3_F * 2 * CNN1D_CONV_3_W];
    reverse_im2col_and_flatten2(conv3_storage_wide, conv3_flatten, CNN1D_CONV_3_F, CNN1D_CONV_3_W, CNN1D_CONV_3_W_conv - 2, 2, WIDE_LEN);
    out_file << std::endl << "Reverse im2col: " << std::endl;
    print_output(conv3_flatten, CNN1D_CONV_3_F, CNN1D_CONV_3_W, out_file);

    out_file << std::endl << "################################## CONV FW 4 ##################################" << std::endl;

    unsigned conv4_valid = 2;
    top_mm_im2col(
    				conv4_storage_wide.data(),
					conv3_storage_wide.data(),
					conv4_w_wide.data(),
					conv4_b.data(),
					conv4_indices.data(),
    				0,
					CNN1D_CONV_4_F, CNN1D_CONV_4_C_im2col, CNN1D_CONV_4_W, CNN1D_CONV_4_W_conv - conv4_valid, 2, 32 / WIDE_LEN, 1, 0, 1, 0, 0, 0, 0
    );

    print_output_wide(conv4_storage_wide.data(), CNN1D_CONV_4_F * 2, 32, WIDE_LEN, out_file);

    HLSNN_DataType *conv4_flatten = new HLSNN_DataType [CNN1D_CONV_4_F * 2 * CNN1D_CONV_4_W];
    reverse_im2col_and_flatten2(conv4_storage_wide, conv4_flatten, CNN1D_CONV_4_F, CNN1D_CONV_4_W, CNN1D_CONV_4_W_conv - conv4_valid, 2, WIDE_LEN);
    out_file << std::endl << "Reverse im2col: " << std::endl;
    print_output(conv4_flatten, CNN1D_CONV_4_F, CNN1D_CONV_4_W, out_file);


    out_file << std::endl << "################################## CONV FW 5 ##################################" << std::endl;

    unsigned conv5_valid = 4;
    top_mm_im2col(
    				conv5_storage_wide.data(),
					conv4_storage_wide.data(),
					conv5_w_wide.data(),
					conv5_b.data(),
					conv5_indices.data(),
    				0,
					CNN1D_CONV_5_F, CNN1D_CONV_5_C_im2col, CNN1D_CONV_5_W, CNN1D_CONV_5_W_conv - conv5_valid, 2, 32 / WIDE_LEN, 0, 0, 1, 0, 1, 0, 0
    );

    print_output_wide(conv5_storage_wide.data(), CNN1D_CONV_5_F, 32, WIDE_LEN, out_file);

    out_file << std::endl << "################################## FC FW 1 ##################################" << std::endl;

    top_mm_im2col(
    				dense1_storage_wide.data(),
					fc1_w_padded_wide.data(),
					conv5_storage_wide.data(),
					fc1_b.data(),
					dense1_indices.data(),
    				0,
    				1, CNN1D_FC_1_C, CNN1D_FC_1_F_widened, 3, 0, 0, 0, 0, 0, 0, 1, 0, 1
    );

    print_flattened_matrix(dense1_storage_wide, CNN1D_FC_1_F_widened, out_file);

    out_file << std::endl << "################################## SOFTMAX ##################################" << std::endl;

    compute_softmax(dense1_storage_wide, label, softmax_storage, 1, 3, WIDE_LEN);

    print_output_wide(softmax_storage.data(), 1, 16, WIDE_LEN, out_file);    

    out_file << std::endl << "################################## dX FC1 ##################################" << std::endl;

    top_mm_im2col(
    				softmax_storage.data(),
					fc1_w_padded_wide.data(),
					d_dense1_storage_wide.data(),
					fc1_b.data(),
					dense1_indices.data(),
    				1,
    				1, 512, 16, 512, 0, 0, 0, 0, 0, 0, 0, 0, 0
    );

    print_output_wide(d_dense1_storage_wide.data(), 1, 512, WIDE_LEN, out_file);

    relu_bw(d_dense1_storage_wide, conv5_storage_wide, 16, 32, WIDE_LEN);
    print_output_wide(d_dense1_storage_wide.data(), 16, 32, WIDE_LEN, out_file);

    out_file << std::endl << "################################## dX CONV5 ##################################" << std::endl;

    top_mm_im2col(
    				d_conv4_storage_wide.data(),
					d_dense1_storage_wide.data(),
					conv5_w_tr_wide.data(),
					conv5_b.data(),
					conv5_indices.data(),
    				0,
					CNN1D_CONV_5_C_im2col, CNN1D_CONV_5_F, 32, 32, 2, 32, 0, 1, 0, 0, 0, 0, 1
    );

    print_output_wide(d_conv4_storage_wide.data(), CNN1D_CONV_5_F, 32, WIDE_LEN, out_file);


    relu_bw_im2col(d_conv4_storage_wide, conv4_flatten, CNN1D_CONV_5_F, 32, 2, 16, WIDE_LEN);
    print_output_wide(d_conv4_storage_wide.data(), 16, 32, WIDE_LEN, out_file);

    out_file << std::endl << "################################## dX CONV4 ##################################" << std::endl;

    top_mm_im2col(
    				d_conv3_storage_wide.data(),
					d_conv4_storage_wide.data(),
					conv4_w_tr_wide.data(),
					conv4_b.data(),
					conv4_indices.data(),
    				0,
					CNN1D_CONV_4_C_im2col, CNN1D_CONV_4_F, 32, 32, 2, 32, 0, 1, 0, 0, 0, 0, 1
    );

    print_output_wide(d_conv3_storage_wide.data(), CNN1D_CONV_4_F, 32, WIDE_LEN, out_file);

    relu_bw_im2col(d_conv3_storage_wide, conv3_flatten, CNN1D_CONV_4_F, 32, 2, 16, WIDE_LEN);
    print_output_wide(d_conv3_storage_wide.data(), 16, 32, WIDE_LEN, out_file);


    out_file << std::endl << "################################## dX CONV3 ##################################" << std::endl;

    top_mm_im2col(
    				d_mp2_storage_wide.data(),
					d_conv3_storage_wide.data(),
					conv3_w_tr_wide.data(),
					conv3_b.data(),
					conv3_indices.data(),
    				0,
					CNN1D_CONV_3_C_im2col, CNN1D_CONV_3_F, 32, 32, 2, 32, 0, 1, 0, 0, 0, 0, 1
    );

    print_output_wide(d_mp2_storage_wide.data(), CNN1D_CONV_3_F, 32, WIDE_LEN, out_file);

    //relu_bw_im2col(d_conv2_storage_wide, conv2_flatten, CNN2_CONV_3_F, 32, 2, 16, 16);
    maxpool_relu_bw_im2col(d_mp2_storage_wide, d_conv2_storage_wide, conv2_flatten, conv2_indices, 32, 64, 3, 32, WIDE_LEN);
    print_output_wide(d_conv2_storage_wide.data(), CNN1D_CONV_3_F, 64, WIDE_LEN, out_file);

    out_file << std::endl << "################################## dX CONV2 ##################################" << std::endl;


    top_mm_im2col(
    				d_mp1_storage_wide.data(),
					d_conv2_storage_wide.data(),
					conv2_w_tr_wide.data(),
					conv2_b.data(),
					conv2_indices.data(),
    				0,
					CNN1D_CONV_2_C_im2col, CNN1D_CONV_2_F, 64, 64, 2, 64, 0, 1, 0, 0, 0, 0, 1
    );

    print_output_wide(d_mp1_storage_wide.data(), 32, 64, WIDE_LEN, out_file);

    maxpool_relu_bw_im2col(d_mp1_storage_wide, d_conv1_storage_wide, mp1_flatten, conv1_indices, 32, 128, 2, 64, WIDE_LEN);
    print_output_wide(d_conv1_storage_wide.data(), CNN1D_CONV_1_F, 128, WIDE_LEN, out_file);

    ///

    out_file << std::endl << "################################## dW FC1 ##################################" << std::endl;

	top_mm_im2col(
				softmax_storage.data(),
				d_fc1_w_wide.data(),
				conv5_storage_wide.data(),
				fc1_b.data(),
				dense1_indices.data(),
				2,
				1, 512, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    );

	print_output_wide(d_fc1_w_wide.data(), 512, 16, WIDE_LEN, out_file);

	out_file << std::endl << "################################## dW CONV 5 ##################################" << std::endl;

    top_mm_im2col(
					conv4_storage_wide.data(),
					d_dense1_storage_wide.data(),
					d_conv5_w_wide.data(),
					conv5_b.data(),
					conv5_indices.data(),
    				1,
					CNN1D_CONV_5_C_im2col, CNN1D_CONV_5_F, CNN1D_CONV_5_W, CNN1D_CONV_5_W, CNN1D_CONV_5_K, CNN1D_CONV_5_F / WIDE_LEN, 0, 0, 0, 0, 0, 0, 0
    );

    print_output_wide(d_conv5_w_wide.data(), CNN1D_CONV_5_C_im2col, CNN1D_CONV_5_F, WIDE_LEN, out_file);

	out_file << std::endl << "################################## dW CONV 4 ##################################" << std::endl;

    top_mm_im2col(
					conv3_storage_wide.data(),
					d_conv4_storage_wide.data(),
					d_conv4_w_wide.data(),
					conv4_b.data(),
					conv4_indices.data(),
    				1,
					CNN1D_CONV_4_C_im2col, CNN1D_CONV_4_F, CNN1D_CONV_4_W, CNN1D_CONV_4_W, CNN1D_CONV_4_K, CNN1D_CONV_4_F / WIDE_LEN, 0, 0, 0, 0, 0, 0, 0
    );

    print_output_wide(d_conv4_w_wide.data(), CNN1D_CONV_4_C_im2col, CNN1D_CONV_4_F, WIDE_LEN, out_file);

	out_file << std::endl << "################################## dW CONV 3 ##################################" << std::endl;
    top_mm_im2col(
					mp2_storage_wide.data(),
					d_conv3_storage_wide.data(),
					d_conv3_w_wide.data(),
					conv3_b.data(),
					conv3_indices.data(),
    				1,
					CNN1D_CONV_3_C_im2col, CNN1D_CONV_3_F, CNN1D_CONV_3_W, CNN1D_CONV_3_W, CNN1D_CONV_3_K, CNN1D_CONV_3_F / WIDE_LEN, 0, 0, 0, 0, 0, 0, 0
    );

    print_output_wide(d_conv3_w_wide.data(), CNN1D_CONV_3_C_im2col, CNN1D_CONV_3_F, WIDE_LEN, out_file);


	out_file << std::endl << "################################## dW CONV 2 ##################################" << std::endl;
    top_mm_im2col(
					mp1_storage_wide.data(),
					d_conv2_storage_wide.data(),
					d_conv2_w_wide.data(),
					conv2_b.data(),
					conv2_indices.data(),
    				1,
					CNN1D_CONV_2_C_im2col, CNN1D_CONV_2_F, CNN1D_CONV_2_W, CNN1D_CONV_2_W, CNN1D_CONV_2_K, CNN1D_CONV_2_F / WIDE_LEN, 0, 0, 0, 0, 0, 0, 0
    );

    print_output_wide(d_conv2_w_wide.data(), CNN1D_CONV_2_C_im2col, CNN1D_CONV_2_F, WIDE_LEN, out_file);

	out_file << std::endl << "################################## dW CONV 1 ##################################" << std::endl;
    top_mm_im2col(
					input_storage_wide.data(),
					d_conv1_storage_wide.data(),
					d_conv1_w_wide.data(),
					conv1_b.data(),
					conv1_indices.data(),
    				1,
					CNN1D_CONV_1_C_im2col_padded, CNN1D_CONV_1_F, CNN1D_CONV_1_W, CNN1D_CONV_1_W, CNN1D_CONV_1_K, CNN1D_CONV_1_F / WIDE_LEN, 0, 0, 0, 0, 0, 0, 0
    );

    print_output_wide(d_conv1_w_wide.data(), CNN1D_CONV_1_C_im2col_padded, CNN1D_CONV_1_F, WIDE_LEN, out_file);

    out_file << std::endl << "################################## WEIGHT UPDATES ##################################" << std::endl;

    float learning_rate = 0.01;

    out_file << std::endl << "################################## WU FC1 ##################################" << std::endl;
    weight_sgd(fc1_w_padded_wide, d_fc1_w_wide, CNN1D_FC_1_C, CNN1D_FC_1_F_widened, learning_rate, WIDE_LEN);
    print_output_wide(fc1_w_padded_wide.data(), CNN1D_FC_1_C, CNN1D_FC_1_F_widened, WIDE_LEN, out_file);

    out_file << std::endl << "################################## WU CONV 5 ##################################" << std::endl;
    weight_sgd(conv5_w_tr_wide, d_conv5_w_wide, CNN1D_CONV_5_C_im2col, CNN1D_CONV_5_F, learning_rate, WIDE_LEN);
    print_output_wide(conv5_w_tr_wide.data(), CNN1D_CONV_5_C_im2col, CNN1D_CONV_5_F, WIDE_LEN, out_file);

    out_file << std::endl << "################################## WU CONV 5 TR ##################################" << std::endl;
    transpose_matrix_widened(conv5_w_tr_wide, conv5_w_wide, CNN1D_CONV_5_C_im2col, CNN1D_CONV_5_F, WIDE_LEN);
    print_output_wide(conv5_w_wide.data(), CNN1D_CONV_5_F, CNN1D_CONV_5_C_im2col, WIDE_LEN, out_file);

    out_file << std::endl << "################################## WU CONV 4 ##################################" << std::endl;
    weight_sgd(conv4_w_tr_wide, d_conv4_w_wide, CNN1D_CONV_4_C_im2col, CNN1D_CONV_4_F, learning_rate, WIDE_LEN);
    print_output_wide(conv4_w_tr_wide.data(), CNN1D_CONV_4_C_im2col, CNN1D_CONV_4_F, WIDE_LEN, out_file);

    out_file << std::endl << "################################## WU CONV 4 TR ##################################" << std::endl;
    transpose_matrix_widened(conv4_w_tr_wide, conv4_w_wide, CNN1D_CONV_4_C_im2col, CNN1D_CONV_4_F, WIDE_LEN);
    print_output_wide(conv4_w_wide.data(), CNN1D_CONV_4_F, CNN1D_CONV_4_C_im2col, WIDE_LEN, out_file);

    out_file << std::endl << "################################## WU CONV 3 ##################################" << std::endl;
    weight_sgd(conv3_w_tr_wide, d_conv3_w_wide, CNN1D_CONV_3_C_im2col, CNN1D_CONV_3_F, learning_rate, WIDE_LEN);
    print_output_wide(conv3_w_tr_wide.data(), CNN1D_CONV_3_C_im2col, CNN1D_CONV_3_F, WIDE_LEN, out_file);

    out_file << std::endl << "################################## WU CONV 3 TR ##################################" << std::endl;
    transpose_matrix_widened(conv3_w_tr_wide, conv3_w_wide, CNN1D_CONV_3_C_im2col, CNN1D_CONV_3_F, WIDE_LEN);
    print_output_wide(conv3_w_wide.data(), CNN1D_CONV_3_F, CNN1D_CONV_3_C_im2col, WIDE_LEN, out_file);

    out_file << std::endl << "################################## WU CONV 2 ##################################" << std::endl;
    weight_sgd(conv2_w_tr_wide, d_conv2_w_wide, CNN1D_CONV_2_C_im2col, CNN1D_CONV_2_F, learning_rate, WIDE_LEN);
    print_output_wide(conv2_w_tr_wide.data(), CNN1D_CONV_2_C_im2col, CNN1D_CONV_2_F, WIDE_LEN, out_file);

    out_file << std::endl << "################################## WU CONV 2 TR ##################################" << std::endl;
    transpose_matrix_widened(conv2_w_tr_wide, conv2_w_wide, CNN1D_CONV_2_C_im2col, CNN1D_CONV_2_F, WIDE_LEN);
    print_output_wide(conv2_w_wide.data(), CNN1D_CONV_2_F, CNN1D_CONV_2_C_im2col, WIDE_LEN, out_file);

    out_file << std::endl << "################################## WU CONV 1 ##################################" << std::endl;
    weight_sgd(conv1_w_tr_wide, d_conv1_w_wide, CNN1D_CONV_1_C_im2col, CNN1D_CONV_1_F, learning_rate, WIDE_LEN);
    print_output_wide(conv1_w_tr_wide.data(), CNN1D_CONV_1_C_im2col, CNN1D_CONV_1_F, WIDE_LEN, out_file);

    out_file << std::endl << "################################## WU CONV 1 TR ##################################" << std::endl;
    transpose_matrix_widened(conv1_w_tr_wide, conv1_w_wide, CNN1D_CONV_1_C_im2col_padded, CNN1D_CONV_1_F, WIDE_LEN);
    print_output_wide(conv1_w_wide.data(), CNN1D_CONV_1_F, CNN1D_CONV_1_C_im2col_padded, WIDE_LEN, out_file);

    out_file << std::endl << "################################## BIAS UPDATES ##################################" << std::endl;

    //revise bias
    out_file << "FC1 b" << std::endl;
    bias_sgd_FC(fc1_b, softmax_storage, 1, CNN1D_FC_1_F_widened, CNN1D_FC_1_F, learning_rate, WIDE_LEN);
    print_vector(fc1_b, CNN1D_FC_1_F_widened, out_file);
    out_file << std::endl;

    out_file << "Conv 5 b" << std::endl;
    do_dB_conv(d_dense1_storage_wide, conv5_b, CNN1D_CONV_5_F, CNN1D_CONV_5_F, CNN1D_CONV_5_W, learning_rate, WIDE_LEN);
    print_vector(conv5_b, CNN1D_CONV_5_F, out_file);
    out_file << std::endl;

    out_file << "Conv 4 b" << std::endl;
    do_dB_conv(d_conv4_storage_wide, conv4_b, CNN1D_CONV_4_F, CNN1D_CONV_4_F, CNN1D_CONV_4_W, learning_rate, WIDE_LEN);
    print_vector(conv4_b, CNN1D_CONV_4_F, out_file);
    out_file << std::endl;

    out_file << "Conv 3 b" << std::endl;
    do_dB_conv(d_conv3_storage_wide, conv3_b, CNN1D_CONV_3_F, CNN1D_CONV_3_F, CNN1D_CONV_3_W, learning_rate, WIDE_LEN);
    print_vector(conv3_b, CNN1D_CONV_3_F, out_file);
    out_file << std::endl;

    out_file << "Conv 2 b" << std::endl;
    do_dB_conv(d_conv2_storage_wide, conv2_b, CNN1D_CONV_2_F, CNN1D_CONV_2_F, CNN1D_CONV_2_W, learning_rate, WIDE_LEN);
    print_vector(conv2_b, CNN1D_CONV_2_F, out_file);
    out_file << std::endl;

    out_file << "Conv 1 b" << std::endl;
    do_dB_conv(d_conv1_storage_wide, conv1_b, CNN1D_CONV_1_F, CNN1D_CONV_1_F, CNN1D_CONV_1_W, learning_rate, WIDE_LEN);
    print_vector(conv1_b, CNN1D_CONV_1_F, out_file);
    out_file << std::endl;

    std::cout << "End test 2" << std::endl;

}


int main(){

	std::cout << "Start test bench file" << std::endl;

	test_1dcnn_pipeline();

    return 0;
}

