#include <cmath>
#include <vector>
#include "types.hpp"

struct TANH_thread_data_wide{
    wide_type* in;
    wide_type* out;
    unsigned length = 0;
    unsigned start = 0;
    unsigned end = 0;
    unsigned wide_length = 0;
};

struct TANH_BW_thread_data_wide{
    wide_type* in;
    wide_type* grad;
    wide_type* out;
    unsigned length = 0;
    unsigned start = 0;
    unsigned end = 0;
    unsigned wide_length = 0;
};

void *worker_TANH_wide(void *args){

    struct TANH_thread_data_wide *args_ = ((struct TANH_thread_data_wide*) args);

    wide_type* in = args_->in;
    wide_type* out = args_->out;
    unsigned start = args_->start;
    unsigned end = args_->end;
    unsigned wide_length = args_->wide_length;

    unsigned f = 0;
    for(f = start; f < end; f++){
        for(unsigned j = 0; j < wide_length; j++){
            out[f][j] = HLSNN_DataType(std::tanh(float(in[f][j])));
        }
    }

}

void *worker_TANH_BW_wide(void *args){

    struct TANH_BW_thread_data_wide *args_ = ((struct TANH_BW_thread_data_wide*) args);

    wide_type* in = args_->in;
    wide_type* grad = args_->grad;
    wide_type* out = args_->out;
    unsigned start = args_->start;
    unsigned end = args_->end;
    unsigned wide_length = args_->wide_length;

    unsigned f = 0;
    for(f = start; f < end; f++){
        for(unsigned j = 0; j < wide_length; j++){
            float tanh_v = float(in[f][j]);
            float tanh_pow = 1.0 - (tanh_v * tanh_v);
            float grad_ = float(grad[f][j]);
            out[f][j] = grad_ * tanh_pow;

        }
    }

}

void tanh_activation(
    wide_type *A,
    wide_type *B,
    int rows,
    int cols,
    unsigned wide_length
) {
    int total_elements = rows * cols;
    for (int i = 0; i < total_elements; ++i) {
        int chunk_index = i / wide_length;
        int offset = i % wide_length;
        B[chunk_index][offset] = HLSNN_DataType(std::tanh(float(A[chunk_index][offset])));
    }
}


void tanh_activation_bw(
    wide_type *A,
    wide_type *B,
    wide_type *GRAD,
    int rows,
    int cols,
    unsigned wide_length
) {
    int total_elements = rows * cols;
    for (int i = 0; i < total_elements; ++i) {
        int chunk_index = i / wide_length;
        int offset = i % wide_length;
        
        float tanh_v = float(B[chunk_index][offset]);
        
        float tanh_pow = 1.0 - (tanh_v * tanh_v);
        float grad = float(GRAD[chunk_index][offset]);
        A[chunk_index][offset] = grad * tanh_pow;
    }
}


typedef float math_type;

template<unsigned B_, unsigned F_>
static float softmax_array_2(
        wide_type *inM,
        wide_type *outD,
        float *labels,
		unsigned B, unsigned F, unsigned wide_col, unsigned wide_length){

	math_type expsum_tmp[B_][F_];
    math_type epsilon = 0.00000001;
    math_type softmax_tmp[B_][F_];

	float loss;
	float expsum[B_];
    unsigned chunk_index = 0;
    for(int i = 0; i < B; i++){
    	math_type tmp = math_type(0.0);
        unsigned col_index = 0;
        for(int j = 0; j < wide_col; j++){
            for(int jj = 0; jj < wide_length; jj++){
                if(col_index < F){
        		    math_type v = (math_type) inM[chunk_index][col_index] + epsilon;
                    tmp += (std::exp(v));
        	    }
                col_index++;
            }
        }
        chunk_index++;
        expsum[i] = tmp;
    }


    float sum = 0.0;
    chunk_index = 0;
    for(int i = 0; i < B; i++){
        unsigned col_index = 0;
        for(int j = 0; j < wide_col; j++){
            for(int jj = 0; jj < wide_length; jj++){
                if(col_index < F){
                    float v = (float)inM[chunk_index][col_index];
                    softmax_tmp[i][col_index] = std::exp(v) / expsum[i];
                    sum += softmax_tmp[i][col_index];
                }
                else{
                    softmax_tmp[i][col_index] = float(0.0);
                }
                col_index++;
            }
        }
        chunk_index++;
    }

    // This is only for calculating accuracy and loss
    unsigned argmaxes[B_];
    for(int i = 0; i < B; i++){
        math_type max = 0.0;
        unsigned argmax = 0;
    	for(int j = 0; j < F; j++){
            if(i < B && j < F){
                if(max < float(softmax_tmp[i][j])){
                    max = float(softmax_tmp[i][j]);
                    argmax = j;
                }
            }
        }
        argmaxes[i] = argmax;        
    }    

    float acc = 0.0;
    float loss_sum = 0.0;
    unsigned vector_index = 0;
    for(int i = 0; i < B; i++){
    	for(int j = 0; j < F; j++){
            unsigned argmax = 0;
            float max_ = 0.0;
        	if(i < B && j < F){
                if(float(labels[vector_index]) > 0.5){
                    loss_sum += -1 * std::log(std::max(float(softmax_tmp[i][j]) * float(labels[vector_index]), epsilon));

                    if(argmaxes[i] == j){
                        acc += 1.0;
                    }
                }

                vector_index++;
        	}
        	else{
        	}
        }
    }

    unsigned c = 0;
    chunk_index = 0;
    vector_index = 0;
    for(int i = 0; i < B; i++){
        unsigned col_index = 0;
    	for(int j = 0; j < wide_col; j++){
            for(int jj = 0; jj < wide_length; jj++){
                if(col_index < F){
                    outD[chunk_index][col_index] = HLSNN_DataType((softmax_tmp[i][col_index] - labels[vector_index]) / B);
                    vector_index++;
                }
                else{
                    outD[chunk_index][col_index] = HLSNN_DataType(0.0);
                }
                col_index++;
            }
        }
        chunk_index++;
    }

    return acc;
}
