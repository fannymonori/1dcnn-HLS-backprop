#include <vector>
#include "types.hpp"

// ====================================================================================================================
// WEIGHT UPDATE

void weight_sgd(wide_type* matrix, wide_type* gradient, int M, int N, float learning_rate, unsigned wide_length) {
    int total_elements = M * N;
    for (int i = 0; i < total_elements; i++) {
        int tile_index = i / wide_length;
        int tile_element = i % wide_length;
        matrix[tile_index][tile_element] = HLSNN_DataType(float(matrix[tile_index][tile_element]) - float(learning_rate) * float(gradient[tile_index][tile_element]));
    }
}

// ====================================================================================================================
// WEIGHT UPDATE MULTI-THREADING

struct SGD_thread_data_wide{
    wide_type* w_orig;
    wide_type* w;
    float LR = 0.001;
    unsigned length = 0;
    unsigned start = 0;
    unsigned end = 0;
    unsigned wide_length = 0;
};

/*
This function is for performing one step of SGD parameter update on wide data type
*/
void *worker_SGD_wide(void *args){

    struct SGD_thread_data_wide *args_ = ((struct SGD_thread_data_wide*) args);

    float learning_rate = args_->LR;

    wide_type* w_orig = args_->w_orig;
    wide_type* w = args_->w;
    unsigned start = args_->start;
    unsigned end = args_->end;
    unsigned wide_length = args_->wide_length;

    unsigned f = 0;
    for(f = start; f < end; f++){
        for(unsigned j = 0; j < wide_length; j++){
            w_orig[f][j] = float(w_orig[f][j]) - learning_rate * float(w[f][j]);
        }
    }

}

// ====================================================================================================================
// BIAS dB and UPDATE

void bias_sgd_FC(HLSNN_DataType* matrix, wide_type* gradient, unsigned M, unsigned N, unsigned N_orig, float learning_rate, unsigned wide_length) {
    HLSNN_DataType lr_ = HLSNN_DataType(learning_rate);

	unsigned loop_end = N / wide_length < 1 ? 1 : N / wide_length;

	if(N % wide_length != 0){
		loop_end++;
	}

	unsigned index = 0;
	unsigned flat_index = 0;
	for(int i = 0; i < M; i++){
		unsigned col_index = 0;
		for(int j = 0; j < loop_end; j++){
			for(unsigned jj = 0; jj < wide_length; jj++){
				if(col_index < N_orig){
					matrix[flat_index] = matrix[flat_index] - lr_ * gradient[index][jj];
				}
				else{
					matrix[flat_index] = 0.0;
				}
				flat_index++;
				col_index++;
			}
			index++;
		}
	}
}

void do_dB_conv(wide_type* dx, HLSNN_DataType* b, unsigned M, unsigned M_nonw, unsigned N, float learning_rate,unsigned wide_len){
    HLSNN_DataType lr_ = HLSNN_DataType(learning_rate);
	unsigned loop_end = N / wide_len < 1 ? 1 : N / wide_len;

	std::vector<HLSNN_DataType> b_grad;
	b_grad.resize(M, 0.0);

	unsigned index = 0;
	for(int i = 0; i < M; i++){
        unsigned col_index = 0;
		for(int j = 0; j < loop_end; j++){
			for(int jj = 0; jj < wide_len; jj++){
				b_grad[i] += dx[index][jj];
                col_index++;
			}
			index++;
		}
	}

	for(int i = 0; i < M; i++){
		if(i < M_nonw){
			b[i] = b[i] - lr_ * b_grad[i];
		}
		else{
			b[i] = 0.0;
		}
	}

}