#include "types.hpp"

/**
  Worker function for computing dX of MP layers, where the result was stored in the usual format (not im2col).
*/
void maxpool_bw(wide_type* gradient_in, wide_type* gradient_out, unsigned* indices, int M, int N, int wide_length) {

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

struct mp_relu_bw_thread_data{
    wide_type* gradient_in;
    wide_type* gradient_out;
    HLSNN_DataType* mp_out;
    unsigned* indices;
    unsigned M;
    unsigned N;
    unsigned K;
    unsigned C;
    unsigned wide_length;

    unsigned start;
    unsigned end;
};

/**
 Worker function for computing dX of ReLU+MP layers, where the result was stored in im2col format. This is for thread execution.
*/
void *worker_maxpool_relu_bw_im2col(void *args){

    struct mp_relu_bw_thread_data *args_ = ((struct mp_relu_bw_thread_data*) args);

    unsigned p = 2;    

    wide_type* gradient_in = args_->gradient_in;
    wide_type* gradient_out = args_->gradient_out;
    HLSNN_DataType* mp_out = args_->mp_out;
    unsigned* indices = args_->indices;

    unsigned M = args_->M;
    unsigned N = args_->N;
    unsigned K = args_->K;
    unsigned C = args_->C;
    unsigned wide_length = args_->wide_length;

    unsigned start = args_->start;
    unsigned end = args_->end;

    unsigned loop_end = N / wide_length < 1 ? 1 : N / wide_length;
	unsigned diff = C - (N/p);
	unsigned index = 0;
	unsigned new_index = 0;
	unsigned indices_index = 0;
	unsigned mp_index = 0;
	unsigned flat_index = 0;
	index = 0;

    unsigned th = 0;
    for(th = start; th < end; th++){

        unsigned index = th * loop_end;
        unsigned indices_index = th * (N/p);
        unsigned mp_index = th * (N/p);
        unsigned flat_index = th * (N/p);

        for(int j = 0; j < loop_end; j++){
			for(unsigned jj = 0; jj < wide_length; jj=jj+p){
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
 Function for computing dX of ReLU+MP layers, where the result was stored in im2col format.
*/
void maxpool_relu_bw_im2col(wide_type* gradient_in, wide_type* gradient_out, HLSNN_DataType *mp_out, unsigned* indices, int M, int N, unsigned K, int C, int wide_length) {

    unsigned p = 2;  

	unsigned loop_end = N / wide_length < 1 ? 1 : N / wide_length;
	unsigned N_2 = N / p;
	unsigned diff = C - N_2;
	unsigned loop_end2 = N_2 / wide_length < 1 ? 1 : N_2 / wide_length;
	unsigned index = 0;

	if(N_2 % wide_length != 0){
		loop_end2++;
	}

	unsigned new_index = 0;

	unsigned indices_index = 0;
	unsigned mp_index = 0;
	unsigned flat_index = 0;
	index = 0;
	for(int i = 0; i < M; i++){
		for(int j = 0; j < loop_end; j++){
			for(unsigned jj = 0; jj < wide_length; jj=jj+p){
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
  Function for computing dX of ReLU+MP layers, where the result was stored in the usual format (not im2col).
*/
void maxpool_relu_bw(wide_type* gradient_in, wide_type* gradient_out, wide_type* mp_out, unsigned* indices, int M, int N, int wide_length) {

    unsigned p = 2; 

	unsigned loop_end = N / wide_length < 1 ? 1 : N / wide_length;
	unsigned N_2 = N / p;

	unsigned index = 0;
	unsigned indices_index = 0;

	for(int i = 0; i < M; i++){
		unsigned col_index = 0;
		for(int j = 0; j < loop_end; j++){
			for(unsigned k = 0; k < wide_length; k=k+p){
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
  Function for computing dX of ReLU layers, where the result was stored in the usual format (not im2col).
*/
void relu_bw(wide_type* gradient, wide_type* activation, int M, int N, int wide_length) {

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
  Worker function for computing dX of ReLU layers, where the result was stored in im2col format.
*/
void relu_bw_im2col(wide_type* gradient_out, HLSNN_DataType* activation_in, int M, int N, unsigned K, int C, int wide_length) {
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

struct relu_bw_simple_thread_data{
    wide_type* gradient;
    wide_type* activation;
    unsigned M;
    unsigned N;
    unsigned wide_length;

    unsigned start;
    unsigned end;
};

/**
  Worker function for computing dX of ReLU layers, where the result was stored in the usual format (not im2col). This is for multithreading.
*/
void *worker_relu_bw(void *args){

    struct relu_bw_simple_thread_data *args_ = ((struct relu_bw_simple_thread_data*) args);

    wide_type* gradient = args_->gradient;
    wide_type* activation = args_->activation;

    unsigned M = args_->M;
    unsigned N = args_->N;
    unsigned wide_length = args_->wide_length;

    unsigned start = args_->start;
    unsigned end = args_->end;

    unsigned loop_end = N / wide_length < 1 ? 1 : N / wide_length;

    for (int i = start; i < end; i++) {
        unsigned index = i * loop_end;
        for (unsigned j = 0; j < loop_end; j++) {
            for (unsigned k = 0; k < wide_length; k++) {
                if (activation[index][k] == HLSNN_DataType(0.0)) {
                    gradient[index][k] = HLSNN_DataType(0.0);
                }
            }
            index++;
        }
    }

}

struct relu_bw_thread_data{
    wide_type* gradient_out;
    HLSNN_DataType* activation_in;
    unsigned M;
    unsigned N;
    unsigned K;
    unsigned C;
    unsigned wide_length;

    unsigned start;
    unsigned end;
};

/**
  Worker function for computing dX of ReLU layers, where the result was stored in im2col format. This is for multithreading.
*/
void *worker_relu_bw_im2col(void *args){

    struct relu_bw_thread_data *args_ = ((struct relu_bw_thread_data*) args);

    wide_type* gradient_out = args_->gradient_out;
    HLSNN_DataType* activation_in = args_->activation_in;

    unsigned M = args_->M;
    unsigned N = args_->N;
    unsigned K = args_->K;
    unsigned C = args_->C;
    unsigned wide_length = args_->wide_length;

    unsigned start = args_->start;
    unsigned end = args_->end;

    unsigned loop_end = N / wide_length < 1 ? 1 : N / wide_length;
    unsigned th = 0;
    for(th = start; th < end; th++){
        unsigned index = th * loop_end;
        unsigned flat_index = th * N;
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
