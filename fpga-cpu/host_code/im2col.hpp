#include "types.hpp"

void reverse_mp_im2col_and_flatten(wide_type* mp_im2col, HLSNN_DataType *flat_mp_out, int M, int N, int N_mp_wide, unsigned K, int wide_length) {

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

	//Flatten the widened maxpool buffer to a vector
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

void reverse_conv_im2col_and_flatten(wide_type* im2col, HLSNN_DataType *flat_out, int M, int N, int N_mp_wide, unsigned K, int wide_length) {

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

	//Flatten the widened buffer to a vector
	for(int i = 0; i < M; i++){
		for(int k = 0; k < K; k++){
			for(int j = 0; j < loop_end; j++){
				for(unsigned jj = 0; jj < wide_length; jj++){
					flat[flat_index] = im2col[index][jj];
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
					flat_out[flat_index_2] = flat[flat_index];
					flat_index++;
					flat_index_2++;
			}
		}
	}
}
