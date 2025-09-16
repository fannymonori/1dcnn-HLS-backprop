#include "types.hpp"

struct dB_thread_data{
    wide_type* ofm;
    HLSNN_DataType* b_grad;
    unsigned C;
    unsigned W;
    unsigned wide_length;
    unsigned start;
    unsigned end;
};

void do_dB(wide_type* ofm, HLSNN_DataType* b_grad, unsigned M, unsigned N, unsigned wide_len){
	unsigned c = 0;
	unsigned loop_end = N / wide_len < 1 ? 1 : N / wide_len;

	for(int i = 0; i < M; i++){
        unsigned index = 0;
		for(int j = 0; j < loop_end; j++){

			wide_type tmp = ofm[c];
			for(int k = 0; k < wide_len; k++){
                b_grad[index] += tmp[k];
                index++;
			}
			c++;
		}
	}
}

void *worker_do_dB_conv(void *args){
    struct dB_thread_data *args_ = ((struct dB_thread_data*) args);

    wide_type* ofm = args_->ofm;
    HLSNN_DataType* b_grad = args_->b_grad;
    unsigned C = args_->C;
    unsigned W = args_->W;
    unsigned wide_length = args_->wide_length;
    unsigned start = args_->start;
    unsigned end = args_->end;

	unsigned c = 0;
	unsigned loop_end = W / wide_length < 1 ? 1 : W / wide_length;

	for(int i = start; i < end; i++){
        unsigned index = 0;
		for(int j = 0; j < loop_end; j++){
			for(int k = 0; k < wide_length; k++){
                b_grad[index] += ofm[c][k];
                index++;
			}
			c++;
		}
	}
}

struct dB_SGD_thread_data{
    wide_type* ofm;
    HLSNN_DataType* b;
    unsigned C;
    unsigned C_orig;
    unsigned W;
    float learning_rate;
    unsigned wide_length;
    unsigned start;
    unsigned end;
};

void *worker_do_dB_SGD_conv(void *args){
    struct dB_SGD_thread_data *args_ = ((struct dB_SGD_thread_data*) args);

    wide_type* ofm = args_->ofm;
    HLSNN_DataType* b = args_->b;
    unsigned C = args_->C;
    unsigned C_orig = args_->C_orig;
    unsigned W = args_->W;
    float learning_rate = args_->learning_rate;
    unsigned wide_length = args_->wide_length;
    unsigned start = args_->start;
    unsigned end = args_->end;

    HLSNN_DataType lr_ = HLSNN_DataType(learning_rate);
    unsigned loop_end = W / wide_length < 1 ? 1 : W / wide_length;

    for (unsigned i = start; i < end; i++) {
        HLSNN_DataType acc = 0.0;
        unsigned index = i * loop_end;
        for (unsigned j = 0; j < loop_end; j++) {
            for (unsigned jj = 0; jj < wide_length; jj++) {
                acc += ofm[index][jj];
            }
            index++;
        }

        if (i < C_orig) {
            b[i] = b[i] - lr_ * acc;
        } else {
            b[i] = 0.0;
        }
    }
}
