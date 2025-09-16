#include <vector>
#include "types.hpp"

#define NUM_THREADS 4

void weight_sgd(wide_type *matrix, wide_type *grad, int M, int N, unsigned wide_length, float learning_rate) {
    int total_elements = M * N;
    for (int i = 0; i < total_elements; ++i) {
        int tile_index = i / wide_length;
        int tile_element = i % wide_length;
        matrix[tile_index][tile_element] = HLSNN_DataType(float(matrix[tile_index][tile_element]) - float(learning_rate) * float(grad[tile_index][tile_element]));
    }
}

std::vector<HLSNN_DataType> sumAxis0(wide_type *tiles, int rows, int cols, int wide_length) {
    std::vector<HLSNN_DataType> result(cols, 0.0);

    for (int i = 0; i < rows; ++i) {
        int row_offset = i * cols;
        for (int j = 0; j < cols; ++j) {
            int idx = row_offset + j;
            result[j] += tiles[idx / wide_length][idx % wide_length];
        }
    }

    return result;
}


void bias_sgd(HLSNN_DataType *vec1, HLSNN_DataType *vec2, unsigned length, float learning_rate) {
    HLSNN_DataType lr_ = HLSNN_DataType(learning_rate);
    for (unsigned i = 0; i < length; i++) {
        vec1[i] = vec1[i] - lr_ * vec2[i];
        vec2[i] = 0.0;
    }
}


//////////////////////////

constexpr int TILE_SIZE = 8;

struct dB_thread_data {
    wide_type* tiles;
    int rows;
    int cols;
    int start_row;
    int end_row;
    std::vector<HLSNN_DataType>* grad_result;
};

void* threadFunc(void* arg) {
    dB_thread_data* args = (dB_thread_data*)arg;
    std::vector<HLSNN_DataType>& grad = *(args->grad_result);
    grad.resize(args->cols, 0.0);

    for (int i = args->start_row; i < args->end_row; ++i) {
        int row_offset = i * args->cols;
        for (int j = 0; j < args->cols; ++j) {
            int idx = row_offset + j;
            grad[j] += args->tiles[idx / TILE_SIZE][idx % TILE_SIZE];
        }
    }
    return nullptr;
}

void setup_db_thread_args(dB_thread_data args[NUM_THREADS], std::vector<HLSNN_DataType> grads[NUM_THREADS], wide_type* tiles, int rows, int cols) {
    int rows_per_thread = rows / NUM_THREADS;
    int extra = rows % NUM_THREADS;

    int start = 0;
    for (int t = 0; t < NUM_THREADS; ++t) {
        int end = start + rows_per_thread + (t < extra ? 1 : 0);

        args[t].tiles = tiles;
        args[t].cols = cols;
        args[t].start_row = start;
        args[t].end_row = end;
        args[t].grad_result = &grads[t];

        start = end;
    }
}

std::vector<HLSNN_DataType> sumAxis0_thread(dB_thread_data args[NUM_THREADS]) {
    std::vector<HLSNN_DataType> result(args[0].cols, 0.0);
    pthread_t threads[NUM_THREADS];

    for (int t = 0; t < NUM_THREADS; ++t) {
        pthread_create(&threads[t], nullptr, threadFunc, &args[t]);
    }

    for (int t = 0; t < NUM_THREADS; ++t) {
        pthread_join(threads[t], nullptr);
        for (int j = 0; j < args[t].cols; ++j) {
            result[j] += (*args[t].grad_result)[j];
        }
    }

    return result;
}

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

    //float learning_rate = args_->LR;
    float learning_rate = 0.001;

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
