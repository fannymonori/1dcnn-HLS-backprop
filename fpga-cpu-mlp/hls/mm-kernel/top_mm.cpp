#include "ap_fixed.h"
#include "types.hpp"
#include "hls_math.h"
#include "hls_burst_maxi.h"

#include <math.h>

//#define TILE_SIZE 16
#define TILE_SIZE 8

#define MAX_B 32
#define MAX_F 128
#define MAX_C 128
#define MAX_OUT 6

#define BRAM_BUFFER_SIZE 500
#define BRAM_BUFFER_SIZE_LARGE 1500

#define LR 0.001

extern "C"{
void top_mm(
		hls::burst_maxi<wide_type> out_grad,
		hls::burst_maxi<wide_type> weight,
		hls::burst_maxi<wide_type> input_grad,
		HLSNN_DataType* bias,
		unsigned mode,
		unsigned int B, unsigned int C, unsigned int F,
        unsigned do_tanh
){


#pragma HLS INTERFACE m_axi port=out_grad offset=slave bundle=gmem2 depth=4096

#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem1 depth=1024

#pragma HLS INTERFACE m_axi port=input_grad offset=slave bundle=gmem3 depth=4096

#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem4 depth=128

#pragma HLS INTERFACE s_axilite port=out_grad bundle=control
#pragma HLS INTERFACE s_axilite port=weight bundle=control
#pragma HLS INTERFACE s_axilite port=input_grad bundle=control
#pragma HLS INTERFACE s_axilite port=bias bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=F bundle=control
#pragma HLS INTERFACE s_axilite port=mode bundle=control
#pragma HLS INTERFACE s_axilite port=do_tanh bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    HLSNN_DataType weight_buffer[TILE_SIZE][TILE_SIZE][BRAM_BUFFER_SIZE];
#pragma HLS ARRAY_PARTITION variable=weight_buffer dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight_buffer dim=2 complete
#pragma HLS ARRAY_PARTITION variable=weight_buffer dim=3 off
#pragma HLS bind_storage variable=weight_buffer type=RAM_2P impl=bram

    HLSNN_DataType output_grad_storage[BRAM_BUFFER_SIZE_LARGE];

	HLSNN_DataType bias_storage[MAX_F];
    HLSNN_DataType result_storage[BRAM_BUFFER_SIZE_LARGE];

	HLSNN_DataType X_[TILE_SIZE][TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=X_ dim=0 complete

	HLSNN_DataType A_[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=A_ dim=0 complete

	HLSNN_DataType C_[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=C_ dim=0 complete

	HLSNN_DataType X2[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=X2 dim=0 complete

    wide_type weight_tmp_buffer[BRAM_BUFFER_SIZE];

    //######################################################################################### FORWARD

	bool doRelu = false;
	unsigned weight_offset = 0;
	unsigned bias_offset = 0;
	unsigned output_offset = 0;

	unsigned weight_count = 0;
	unsigned in_count = 0;
	unsigned out_count = 0;

	unsigned weight_pos = 0;


	bool do_FW = 1;
	bool do_dW = 0;
	bool do_dX = 0;

	unsigned C_loop, F_loop;

	if(mode == 0){
		do_FW = 1;
	}
	else if(mode == 1){
		do_FW = 0;
		do_dX = 1;
		do_FW = 0;
	}
	else{
		do_FW = 0;
		do_dW = 1;
		do_dX = 0;
	}


	if(do_FW || do_dX){
        C_loop = C > TILE_SIZE ? (C / TILE_SIZE) : 1;
        F_loop = F > WIDE_LEN ? (F / WIDE_LEN) : 1;
	}
	else{
        C_loop = C > TILE_SIZE ? (C / TILE_SIZE) : 1;
		F_loop = F > TILE_SIZE ? (F / TILE_SIZE) : 1;
	}

	unsigned COLS = F/WIDE_LEN > 0 ?  F/WIDE_LEN : 1;
	unsigned ROWS = C/WIDE_LEN > 0 ?  C/WIDE_LEN : 1;

	for(unsigned b = 0; b < B; b++){
#pragma HLS LOOP_TRIPCOUNT min=1 max=32

		unsigned batch_start_C = b * C;
		unsigned batch_start_F = b * F;

		for (unsigned ff = 0; ff < MAX_F; ff++) {
	#pragma HLS PIPELINE II=1
			bias_storage[ff] = bias[ff];
		}

		if(do_FW || do_dW){
			input_grad.read_request(b * ROWS, ROWS);
			for (unsigned cc = 0; cc < ROWS; cc++) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=128
		#pragma HLS PIPELINE II=1

				wide_type tmp = input_grad.read();
				for(unsigned cc_ = 0; cc_ < WIDE_LEN; cc_++){
					result_storage[cc_ + cc * WIDE_LEN] = tmp[cc_];
				}
			}
		}

		if(do_dX || do_dW){
			out_grad.read_request(b * COLS, COLS);
			for (unsigned f = 0; f < COLS; f++) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=128
	#pragma HLS PIPELINE II=1
				wide_type tmp = out_grad.read();
				for(unsigned ff_ = 0; ff_ < WIDE_LEN; ff_++){
					output_grad_storage[ff_ + f * WIDE_LEN] = tmp[ff_];
				}
			}
		}

		if(do_FW){
			for (unsigned f = 0; f < F; f++) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=128
	#pragma HLS PIPELINE II=1
				output_grad_storage[f] = 0.0;
			}
		}

		if(do_dX){
			for (unsigned f = 0; f < MAX_C; f++) {
	#pragma HLS PIPELINE II=1
				result_storage[f] = 0.0;
			}
		}

	//################## C
		for(unsigned c = 0; c < C_loop; c++){
	#pragma HLS LOOP_TRIPCOUNT min=1 max=8

			unsigned c_offset = c * TILE_SIZE;

			if(do_FW){
				for(unsigned cc = 0; cc < TILE_SIZE; cc++){
					A_[cc] = result_storage[c_offset + cc];
				}
			}

			if(do_dX || do_FW){
				unsigned axi_burst_len = COLS * TILE_SIZE;
				unsigned weight_start = c_offset * (COLS);
				weight.read_request(weight_start, axi_burst_len);
				for (unsigned cc = 0; cc < TILE_SIZE; cc++) { // ROW_TILES

					for (unsigned f_ = 0; f_ < COLS; f_++) { //COLUMNS IN TILES
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
		#pragma HLS PIPELINE II=1
						wide_type tmp = weight.read();
						weight_count++;
						for(unsigned ff = 0; ff < WIDE_LEN; ff++){
		#pragma HLS UNROLL
							weight_buffer[cc][ff][f_] = tmp[ff];
						}
					}
				}
			}

			if(do_dW){
				for(unsigned cc = 0; cc < TILE_SIZE; cc++){
					A_[cc] = result_storage[c_offset + cc];
				}
			}

			for(unsigned ff = 0; ff < TILE_SIZE; ff++){
	#pragma HLS UNROLL
				for(unsigned cc = 0; cc < TILE_SIZE; cc++){
	#pragma HLS UNROLL
					X_[ff][cc] = 0.0;
				}
			}

	//################## F
			for(unsigned f = 0; f < F_loop; f++){
	#pragma HLS LOOP_TRIPCOUNT min=1 max=8
	#pragma HLS PIPELINE II=1
				unsigned f_start = f * TILE_SIZE;


				if(do_FW){
					for(unsigned ff = 0; ff < TILE_SIZE; ff++){
						C_[ff] = output_grad_storage[f_start + ff];
					}
				}

				if(do_dX){
					for(unsigned ff = 0; ff < TILE_SIZE; ff++){
						A_[ff] = output_grad_storage[f_start + ff];
					}

				}

				if(do_dW){

					for(unsigned ff = 0; ff < TILE_SIZE; ff++){
						C_[ff] = output_grad_storage[f_start + ff];
					}
				}

				unsigned tmp_index = 0;


	//################## FF
				for(unsigned ff = 0; ff < TILE_SIZE; ff++){
	#pragma HLS UNROLL

					HLSNN_DataType tmp = C_[ff];

	//################## CC
					for(unsigned cc = 0; cc < TILE_SIZE; cc++){
	#pragma HLS UNROLL

						HLSNN_DataType last = 0.0;
						unsigned A_index = 0;
						HLSNN_DataType B = 0.0;

						if(do_FW){
							A_index = cc;
							last = tmp;
							B = weight_buffer[cc][ff][f];
						}

						if(do_dX){
							A_index = ff;
							last = X_[ff][cc];
							B = weight_buffer[cc][ff][f];
						}

						if(do_dW){
							A_index = cc;
							B = tmp;
							last = 0.0;
						}

						HLSNN_DataType A = A_[A_index];

						HLSNN_DataType X = A * B + last;

						if(do_FW){
							tmp = X;
						}

						if(do_dX){
							X_[ff][cc] = X;
						}

						if(do_dW){
							X_[cc][ff] = X;
						}
					}

					if(do_FW){
						X2[ff] = tmp;
					}

				}

				if(do_dW){

					for(unsigned ff = 0; ff < TILE_SIZE; ff++){
#pragma HLS UNROLL
						for(unsigned cc = 0; cc < TILE_SIZE; cc++){
							weight_buffer[ff][cc][f] = X_ [ff][cc];
						}
					}

				}

				if(do_FW){
					for(unsigned ff = 0; ff < TILE_SIZE; ff++){
						output_grad_storage[f_start + ff] = X2[ff];
					}
				}

			}


			//if(!do_FW){
			if(do_dX){
				for(unsigned i = 0; i < TILE_SIZE; i++){
			#pragma HLS PIPELINE II=1
					HLSNN_DataType tmp = 0;
					for(unsigned j = 0; j < TILE_SIZE; j++){
			#pragma HLS UNROLL
						tmp += X_[j][i];
					}
					result_storage[c_offset + i] = tmp;
				}
			}

			if(do_dW){
				unsigned weight_start = (c_offset) * COLS;
				weight.read_request(weight_start, COLS * TILE_SIZE);

				for(unsigned cc = 0; cc < COLS * TILE_SIZE; cc++){
#pragma HLS LOOP_TRIPCOUNT min=16 max=128
					weight_tmp_buffer[cc] = weight.read();
				}

				weight.write_request(weight_start, COLS * TILE_SIZE);

				unsigned w_tmp_count = 0;
				for(unsigned cc = 0; cc < TILE_SIZE; cc++){

					for(unsigned f_ = 0; f_ < COLS; f_++){
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
						unsigned f_start = f_ * WIDE_LEN;

						wide_type tmp;
						wide_type tmp_2 = weight_tmp_buffer[w_tmp_count];
						for(unsigned ff = 0; ff < WIDE_LEN; ff++){
#pragma HLS UNROLL
							HLSNN_DataType v1 = weight_buffer[cc][ff][f_];
							HLSNN_DataType v2 = tmp_2[ff];
							HLSNN_DataType add_ = v1 + v2;

							tmp[ff] = add_;
						}
						weight.write(tmp);
						w_tmp_count++;
					}
				}
				weight.write_response();
			}

		}


		if(do_FW){

			out_grad.write_request(b * COLS, COLS);

			for(unsigned f = 0; f < COLS; f++){
#pragma HLS LOOP_TRIPCOUNT min=6 max=128
			#pragma HLS PIPELINE II=1

				wide_type tmp;
				for(unsigned ff_ = 0; ff_ < WIDE_LEN; ff_++){
					HLSNN_DataType out = output_grad_storage[ff_ + f * WIDE_LEN];
                    HLSNN_DataType bv = bias_storage[ff_ + f * WIDE_LEN];
                    HLSNN_DataType out_ = out + bv;
                    HLSNN_DataType result = 0.0;

                    if(do_tanh){
                        result = hls::tanh(float(out_));
                    }
                    else{
                        result = out_;
                    }

                    tmp[ff_] = result;
				}

				out_grad.write(tmp);
			}

			out_grad.write_response();
		}

		if(do_dX){
			input_grad.write_request(b * ROWS, ROWS);

			for(unsigned i = 0; i < ROWS; i++){
#pragma HLS LOOP_TRIPCOUNT min=6 max=128
		#pragma HLS PIPELINE II=1
				wide_type tmp;
				for(unsigned cc_ = 0; cc_ < WIDE_LEN; cc_++){
					tmp[cc_] = result_storage[cc_ + i * WIDE_LEN];
				}
				input_grad.write(tmp);
			}

			input_grad.write_response();
		}
	}

}
}