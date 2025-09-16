#include "types.hpp"
#include "hls_math.h"

#define TILE_SIZE 4

extern "C" {
	void tanh_top(
	        wide_type *in,
	        wide_type *out,
            wide_type *grad,
	        int size,
            bool fw
	        )
	{

    #pragma HLS INTERFACE mode=m_axi port=in offset=slave bundle=gmem0
    #pragma HLS INTERFACE mode=m_axi port=out offset=slave bundle=gmem1
    #pragma HLS INTERFACE mode=m_axi port=grad offset=slave bundle=gmem2

    #pragma HLS INTERFACE mode=s_axilite port=in bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=out bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=grad bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=size bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=fw bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=return bundle=control


    HLSNN_DataType in_tile[TILE_SIZE][WIDE_LEN];
    HLSNN_DataType out_tile[TILE_SIZE][WIDE_LEN];
    HLSNN_DataType grad_tile[TILE_SIZE][WIDE_LEN];

    compute_j: for (int j = 0; j < size / WIDE_LEN / TILE_SIZE; j++) {
#pragma HLS loop_tripcount min=16 max=512

            unsigned tile_offset = j * TILE_SIZE;
            
            //READ IN
            for (int t = 0; t < TILE_SIZE; t++) {
#pragma HLS PIPELINE II=1
                wide_type tmp = in[t + tile_offset];
                for (int k = 0; k < WIDE_LEN; k++) {
                    in_tile[t][k] = tmp[k];
                }
            }

            for (int t = 0; t < TILE_SIZE; t++) {
#pragma HLS PIPELINE II=1
                wide_type tmp = grad[t + tile_offset];
                for (int k = 0; k < WIDE_LEN; k++) {
                    grad_tile[t][k] = tmp[k];
                }
            }

            //////////
            compute_t: for (int t = 0; t < TILE_SIZE; t++) {
#pragma HLS PIPELINE II=1
                for(int k = 0; k < WIDE_LEN; k++){
#pragma HLS UNROLL
                    HLSNN_DataType v = in_tile[t][k];
                    HLSNN_DataType result = 0.0;
                    HLSNN_DataType const_1 = 1.0;
                    HLSNN_DataType grad_ = grad_tile[t][k];
                    HLSNN_DataType mult = 0.0;

                    mult = grad_ * (const_1 - (v * v));
                    result = mult;

                    out_tile[t][k] = result;
                }
            }

            //WRITE BACK
            unsigned out_index = 0;
            for (int t = 0; t < TILE_SIZE; t++) {
#pragma HLS PIPELINE II=1
                wide_type out_;
                for (int k = 0; k < WIDE_LEN; k++) {
                    out_[k] = out_tile[t][k];
                }
                out[t + tile_offset] = out_;
            }
        
        
        }
	}
    
}