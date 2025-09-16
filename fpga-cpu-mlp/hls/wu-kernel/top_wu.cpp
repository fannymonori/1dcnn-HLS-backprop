#include <ap_fixed.h>
#include <hls_stream.h>
#include <hls_math.h>
#include "types.hpp"

#define TILE_SIZE 4

#define FLAT_LEN (TILE_SIZE*WIDE_LEN)

extern "C"{
    void top_wu(wide_type *in1, wide_type *in2, unsigned size, float learning_rate){

    #pragma HLS INTERFACE mode=m_axi port=in1 offset=slave bundle=gmem0
    #pragma HLS INTERFACE mode=m_axi port=in2 offset=slave bundle=gmem1

    #pragma HLS INTERFACE mode=s_axilite port=in1 bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=in2 bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=size bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=learning_rate bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control

    const HLSNN_DataType lr = learning_rate;

    HLSNN_DataType w_tile[TILE_SIZE][WIDE_LEN];
    HLSNN_DataType buff1_tile[TILE_SIZE][WIDE_LEN];
    HLSNN_DataType buff_out_tile[TILE_SIZE][WIDE_LEN];

    unsigned tiled_size = size / TILE_SIZE;
    compute_j: for (int j = 0; j < size / WIDE_LEN / TILE_SIZE; ++j) {
#pragma HLS loop_tripcount min=16 max=512

            for (int i = 0; i < TILE_SIZE; i++) {
#pragma HLS PIPELINE II=1
                wide_type tmp = in1[i + j * TILE_SIZE];
                wide_type tmp2 = in2[i + j * TILE_SIZE];
                for (int k = 0; k < WIDE_LEN; k++) {
                    w_tile[i][k] = tmp[k];
                    buff1_tile[i][k] = tmp2[k];
                }
            }

            compute_t: for (int t = 0; t < TILE_SIZE; t++) {
#pragma HLS PIPELINE II=1
                for(int k = 0; k < WIDE_LEN; k++){
#pragma HLS UNROLL
                        HLSNN_DataType lr_ = lr;
                        HLSNN_DataType w_ = w_tile[t][k];
                        HLSNN_DataType grad = buff1_tile[t][k];
                        HLSNN_DataType tmp_out_subtr =  lr_ * grad;
#pragma HLS bind_op variable=tmp_out_subtr op=mul impl=dsp
                        HLSNN_DataType result = w_ - tmp_out_subtr;

                        buff_out_tile[t][k] = result;
                }
            }

            unsigned out_index = 0;
            for (int i = 0; i < TILE_SIZE; i++) {
#pragma HLS PIPELINE II=1
                wide_type out_;
                for (int k = 0; k < WIDE_LEN; k++) {
                    out_[k] = buff_out_tile[i][k];
                }
                in1[i + j * TILE_SIZE] = out_;
            }
        }
    
    }
}

