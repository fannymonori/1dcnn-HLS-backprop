#include "ap_fixed.h"
#include "types.hpp"
#include "hls_math.h"
#include "hls_burst_maxi.h"
#include <math.h>


template<unsigned WIDE_SIZE, class T, unsigned TILE_C, unsigned TILE_F, unsigned F>
void read_weights(
		hls::burst_maxi<wide_type> weight,
		T (&W)[TILE_C][TILE_F][F],
		unsigned COLS,
		unsigned c_offset){

	unsigned axi_burst_len = COLS * TILE_C;
	unsigned weight_start = c_offset * (COLS);

	weight.read_request(weight_start, axi_burst_len);
	for (unsigned cc = 0; cc < TILE_C; cc++) { // ROW_TILES

		for (unsigned f_ = 0; f_ < COLS; f_++) { //COLUMNS IN TILES
#pragma HLS LOOP_TRIPCOUNT min=1 max=8

			wide_type transfer = weight.read();
			for(unsigned ff = 0; ff < WIDE_SIZE; ff++){
				W[cc][ff][f_] = transfer[ff];
			}
		}
	}

}

template<unsigned WIDE_SIZE, class T, unsigned TILE_SIZE, unsigned BUFFER_SIZE, unsigned BRAM_BUFFER_SIZE>
void read_sum_and_write_weights(
		hls::burst_maxi<wide_type> weight,
		T (&BUFFER)[TILE_SIZE][TILE_SIZE][BRAM_BUFFER_SIZE],
		wide_type (&BUFFER_TMP)[BUFFER_SIZE],
		unsigned COLS,
		unsigned c_offset
){
#pragma HLS inline
	unsigned weight_start = (c_offset) * COLS;
	weight.read_request(weight_start, COLS * TILE_SIZE);

	for(unsigned cc = 0; cc < COLS * TILE_SIZE; cc++){
#pragma HLS LOOP_TRIPCOUNT min=16 max=128
		BUFFER_TMP[cc] = weight.read();
	}

	weight.write_request(weight_start, COLS * TILE_SIZE);

	unsigned w_tmp_count = 0;
	for(unsigned cc = 0; cc < TILE_SIZE; cc++){

		for(unsigned f_ = 0; f_ < COLS; f_++){
#pragma HLS LOOP_TRIPCOUNT min=1 max=8

			unsigned f_start = f_ * WIDE_SIZE;

			wide_type tmp;
			wide_type tmp_2 = BUFFER_TMP[w_tmp_count];
			for(unsigned ff = 0; ff < WIDE_SIZE; ff++){
#pragma HLS UNROLL
				HLSNN_DataType v1 = BUFFER[cc][ff][f_];
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


template<class T, unsigned BUFFER_SIZE>
void read_input(
		hls::burst_maxi<wide_type> input,
		T (&IFM)[BUFFER_SIZE],
		unsigned ROWS,
		unsigned b_offset,
		unsigned wide_length){
#pragma HLS inline
	input.read_request(b_offset * ROWS, ROWS);
	for (unsigned cc = 0; cc < ROWS; cc++) {
#pragma HLS LOOP_TRIPCOUNT min=6 max=128

		wide_type tmp = input.read();
		for(unsigned cc_ = 0; cc_ < wide_length; cc_++){
#pragma HLS PIPELINE II=1
			IFM[cc_ + cc * wide_length] = tmp[cc_];
		}
	}

}

template<class T, unsigned BUFFER_SIZE, unsigned BIAS_BUFFER_SIZE>
void write(
		hls::burst_maxi<wide_type> output,
		T (&OFM)[BUFFER_SIZE],
        T (&bias_storage)[BIAS_BUFFER_SIZE],
		unsigned COLS,
		unsigned F,
		unsigned b_offset,
		unsigned wide_length,
		bool do_relu,
        bool do_bias,
		bool fc
		){
#pragma HLS inline
	output.write_request(b_offset * COLS, COLS);

	for(unsigned f = 0; f < COLS; f++){
#pragma HLS LOOP_TRIPCOUNT min=6 max=128
#pragma HLS PIPELINE II=1

		wide_type tmp;
		for(unsigned ff_ = 0; ff_ < wide_length; ff_++){
			unsigned index = ff_ + f * wide_length;
			HLSNN_DataType out = OFM[index];

            HLSNN_DataType b = 0.0;
            
            if(do_bias && fc){
			    b = bias_storage[ff_ + f * WIDE_LEN]; //FC layers
            }

            if(do_bias && !fc){
            	b = bias_storage[b_offset];
            }
			
            HLSNN_DataType out_ = out + b;

			if(index >= F){
				out_ = 0.0;
			}

			if(do_relu && out_ < 0)
				out_ = 0.0;

			tmp[ff_] = out_;
		}

		output.write(tmp);
	}
	output.write_response();
}

template<class T, unsigned BUFFER_SIZE, unsigned bias_size>
void write_im2col(
		hls::burst_maxi<wide_type> output,
		T (&OFM)[BUFFER_SIZE],
		T (&bias_storage)[bias_size],
		unsigned COLS,
		unsigned b_offset,
		unsigned F,
		unsigned K,
		unsigned wide_length,
		unsigned do_relu,
		unsigned do_bias
		){
#pragma HLS inline

	output.write_request(b_offset * (COLS * K), (COLS * K));

	unsigned col_count = 0;
	for(unsigned k = 0; k < K; k++){
		col_count = 0;
        
		for(unsigned f = 0; f < COLS; f++){
#pragma HLS PIPELINE II=1

			wide_type tmp;
			for(unsigned ff_ = 0; ff_ < wide_length; ff_++){
				HLSNN_DataType out = OFM[k + ff_ + f * wide_length];
				HLSNN_DataType b = 0;

				if(do_bias){
					b = bias_storage[b_offset];
				}

				HLSNN_DataType out_ = out + b;

				if((do_relu && out_ < 0) || (col_count >= F - K)){
					out_ = 0.0;
				}

				/*if(col_count >= F - K){
					out_ = 0.0;
				}*/

				tmp[ff_] = out_;
				col_count++;
			}

			output.write(tmp);
		}
	}
	output.write_response();
}

template<unsigned WIDE_SIZE, class T, unsigned BUFFER_SIZE>
void write_col2im(
		hls::burst_maxi<wide_type> output,
		T (&OFM)[BUFFER_SIZE],
		T (&OFM_col2im)[BUFFER_SIZE],
		unsigned &K_count,
		unsigned &out_count,
		unsigned COLS,
		unsigned K
		){
#pragma HLS inline

	for(unsigned f = 0; f < COLS * WIDE_SIZE; f++){
		T tmp = OFM[f];
		T sum = tmp + OFM_col2im[K_count + f];
		OFM_col2im[K_count + f] = sum;
	}

	K_count++;

	if(K_count == K){
		output.write_request(out_count * (COLS), (COLS));
		for(unsigned f = 0; f < COLS; f++){
#pragma HLS LOOP_TRIPCOUNT min=6 max=128
#pragma HLS PIPELINE II=1

			wide_type tmp;
			for(unsigned ff_ = 0; ff_ < WIDE_SIZE; ff_++){
				HLSNN_DataType out = OFM_col2im[ff_ + f * WIDE_SIZE];
				HLSNN_DataType out_ = out;
				tmp[ff_] = out_;
			}
			output.write(tmp);
		}
		output.write_response();

		for(unsigned f = 0; f < COLS * WIDE_SIZE; f++){
			OFM_col2im[f] = 0;
		}

		K_count = 0;
		out_count++;
	}

}

template<class T, unsigned BUFFER_SIZE>
void zero_init_buffer(
		T (&BUFFER)[BUFFER_SIZE]){
#pragma HLS inline
	for (unsigned f = 0; f < BUFFER_SIZE; f++) {
#pragma HLS PIPELINE II=1
		BUFFER[f] = 0.0;
	}
}

template<class T, unsigned TILE_C, unsigned TILE_F>
void zero_init_tile_buffer(
		T (&BUFFER)[TILE_C][TILE_F]){
#pragma HLS inline
	for (unsigned cc = 0; cc < TILE_C; cc++) {
		for (unsigned ff = 0; ff < TILE_F; ff++) {
#pragma HLS UNROLL
			BUFFER[cc][ff] = 0.0;
		}
	}
}

template<class T, unsigned TILE_SIZE, unsigned BUFFER_SIZE>
void read_into_tile_buffer(
		T (&TILE_BUF)[TILE_SIZE],
		T (&BUFFER)[BUFFER_SIZE],
		unsigned buffer_offset
		){
#pragma HLS inline
	for(unsigned tt = 0; tt < TILE_SIZE; tt++){
		TILE_BUF[tt] = BUFFER[buffer_offset + tt];
	}
}

template<class T, unsigned TILE_SIZE, unsigned BUFFER_SIZE>
void read_into_weight_buffer(
		T (&TILE_BUF)[TILE_SIZE][TILE_SIZE],
		T (&BUFFER)[TILE_SIZE][TILE_SIZE][BUFFER_SIZE],
		unsigned f
		){
#pragma HLS inline
	for (unsigned cc = 0; cc < TILE_SIZE; cc++) { // ROW_TILES
		for(unsigned ff = 0; ff < TILE_SIZE; ff++){
			TILE_BUF[cc][ff] = BUFFER[cc][ff][f];
		}
	}
}

template<class T, unsigned TILE_SIZE, unsigned BUFFER_SIZE>
void write_back_to_buffer(
		T (&TILE_BUF)[TILE_SIZE],
		T (&BUFFER)[BUFFER_SIZE],
		unsigned buffer_offset
	){
#pragma HLS inline

	for(unsigned ff = 0; ff < TILE_SIZE; ff++){
		BUFFER[buffer_offset + ff] = TILE_BUF[ff];
	}

}

template<class T, unsigned TILE_SIZE, unsigned BUFFER_SIZE, unsigned BIAS_BUFFER_SIZE>
void write_back_to_buffer(
		T (&TILE_BUF)[TILE_SIZE],
		T (&BUFFER)[BUFFER_SIZE],
        T (&BIAS_BUFFER)[BIAS_BUFFER_SIZE],
		unsigned buffer_offset
	){
#pragma HLS inline
	for(unsigned ff = 0; ff < TILE_SIZE; ff++){
        T b = BIAS_BUFFER[ff];
        BUFFER[buffer_offset + ff] = TILE_BUF[ff] + b;
	}
}

template<class T, unsigned TILE_SIZE>
void write_back_to_2d_buffer(
		T (&TILE_BUF_1)[TILE_SIZE][TILE_SIZE],
		T (&TILE_BUF_2)[TILE_SIZE][TILE_SIZE]
		){
#pragma HLS inline
	for(unsigned ff = 0; ff < TILE_SIZE; ff++){
#pragma HLS UNROLL
		for(unsigned cc = 0; cc < TILE_SIZE; cc++){
			TILE_BUF_1[ff][cc] = TILE_BUF_2[ff][cc];
		}
	}
}

template<class T, unsigned TILE_SIZE, unsigned BUFFER_SIZE>
void write_back_to_weight_buffer(
		T (&TILE_BUF_1)[TILE_SIZE][TILE_SIZE][BUFFER_SIZE],
		T (&TILE_BUF_2)[TILE_SIZE][TILE_SIZE],
		unsigned f
		){
#pragma HLS inline
	for(unsigned ff = 0; ff < TILE_SIZE; ff++){
#pragma HLS UNROLL
		for(unsigned cc = 0; cc < TILE_SIZE; cc++){
			TILE_BUF_1[ff][cc][f] = TILE_BUF_2[ff][cc];
		}
	}
}

template<unsigned TILE_SIZE, unsigned BUFFER_SIZE>
void reduce_and_store(
		HLSNN_DataType (&TILE_BUF)[TILE_SIZE][TILE_SIZE],
		HLSNN_DataType (&BUFFER)[BUFFER_SIZE],
		unsigned buffer_offset
){
	for(unsigned i = 0; i < TILE_SIZE; i++){
#pragma HLS PIPELINE II=1
		HLSNN_DataType tmp = 0;
		for(unsigned j = 0; j < TILE_SIZE; j++){
#pragma HLS UNROLL
			tmp += TILE_BUF[j][i];
		}
		BUFFER[buffer_offset + i] = tmp;
	}
}

/////////////////

#define TILE_SIZE 8

#define LR 0.001

#define URAM_BUFFER_SIZE 10000

#define URAM_BUFFER_SIZE_small 4096

#define BRAM_BUFFER_SIZE 500
#define BRAM_BUFFER_SIZE_LARGE 1500

extern "C" {
void top_mm_im2col(
		hls::burst_maxi<wide_type> out_grad,
		hls::burst_maxi<wide_type> weight,
		hls::burst_maxi<wide_type> input_grad,
		HLSNN_DataType* bias,
		hls::burst_maxi<unsigned> indices,
		unsigned mode,
		unsigned int B,
        unsigned int C,
        unsigned int F,
		unsigned int F_non_padded,
        unsigned int K,
		unsigned int col_out,
		bool do_im2col,
        bool do_col2im,
		bool do_relu,
		bool do_bias_mp,
		bool do_bias_end,
		bool do_maxpool,
		bool do_fc
){

#pragma HLS INTERFACE m_axi port=out_grad offset=slave bundle=gmem2 depth=4160
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem1 depth=4160
#pragma HLS INTERFACE m_axi port=input_grad offset=slave bundle=gmem3 depth=128
#pragma HLS INTERFACE m_axi port=indices offset=slave bundle=gmem4 depth=32
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem5 depth=32


#pragma HLS INTERFACE s_axilite port=out_grad bundle=control
#pragma HLS INTERFACE s_axilite port=weight bundle=control
#pragma HLS INTERFACE s_axilite port=input_grad bundle=control
#pragma HLS INTERFACE s_axilite port=bias bundle=control
#pragma HLS INTERFACE s_axilite port=indices bundle=control
#pragma HLS INTERFACE s_axilite port=mode bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=F bundle=control
#pragma HLS INTERFACE s_axilite port=F_non_padded bundle=control
#pragma HLS INTERFACE s_axilite port=K bundle=control
#pragma HLS INTERFACE s_axilite port=col_out bundle=control
#pragma HLS INTERFACE s_axilite port=do_im2col bundle=control
#pragma HLS INTERFACE s_axilite port=do_col2im bundle=control
#pragma HLS INTERFACE s_axilite port=do_relu bundle=control
#pragma HLS INTERFACE s_axilite port=do_bias_mp bundle=control
#pragma HLS INTERFACE s_axilite port=do_bias_end bundle=control
#pragma HLS INTERFACE s_axilite port=do_maxpool bundle=control
#pragma HLS INTERFACE s_axilite port=do_fc bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control


	// ==== LARGE BUFFERS (BRAM) =========================================================

	HLSNN_DataType weight_buffer[TILE_SIZE][TILE_SIZE][BRAM_BUFFER_SIZE];
#pragma HLS ARRAY_PARTITION variable=weight_buffer dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weight_buffer dim=2 complete
#pragma HLS ARRAY_PARTITION variable=weight_buffer dim=3 off
#pragma HLS bind_storage variable=weight_buffer type=RAM_2P impl=bram

	HLSNN_DataType output_grad_storage[BRAM_BUFFER_SIZE_LARGE];
	HLSNN_DataType output_grad_storage_col2im[BRAM_BUFFER_SIZE_LARGE];
	HLSNN_DataType bias_storage[BRAM_BUFFER_SIZE_LARGE];
	HLSNN_DataType result_storage[BRAM_BUFFER_SIZE];
	HLSNN_DataType input_storage[BRAM_BUFFER_SIZE];

	wide_type weight_tmp_buffer[BRAM_BUFFER_SIZE];

	index_type indices_buffer_maxpool[5000];

	// ==== TILE BUFFERS (REGISTERS) =======================================================
	HLSNN_DataType A_[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=A_ dim=0 complete

	HLSNN_DataType C_[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=C_ dim=0 complete

	HLSNN_DataType W_[TILE_SIZE][TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=W_ dim=1 complete
#pragma HLS ARRAY_PARTITION variable=W_ dim=2 complete

	HLSNN_DataType X2_[TILE_SIZE][TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=X2_ dim=1 complete
#pragma HLS ARRAY_PARTITION variable=X2_ dim=2 complete

	HLSNN_DataType X_[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=X_ dim=0 complete

	// ================================================================================================
	// ================================================================================================

//#pragma HLS DATAFLOW

	bool doRelu = false;
	unsigned weight_offset = 0;
	unsigned bias_offset = 0;
	unsigned output_offset = 0;

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
        F_loop = F > TILE_SIZE ? (F / TILE_SIZE) : 1;
	}
	else{
        C_loop = C > TILE_SIZE ? (C / TILE_SIZE) : 1;
        F_loop = F > TILE_SIZE ? (F / TILE_SIZE) : 1;
	}

	unsigned out_count = 0;
	unsigned K_count = 0;

	unsigned COLS = F/WIDE_LEN > 0 ?  F/WIDE_LEN : 1;
	unsigned ROWS = C/WIDE_LEN > 0 ?  C/WIDE_LEN : 1;

	// ================================================================================================ B
	// START PROCESSING ROWS
	for(unsigned b = 0; b < B; b++){
#pragma HLS LOOP_TRIPCOUNT min=1 max=32

		unsigned batch_start_C = b * C;
		unsigned batch_start_F = b * F;

		for (unsigned ff = 0; ff < F; ff++) {
	#pragma HLS PIPELINE II=1
			bias_storage[ff] = bias[ff];
		}

		// INPUT READ
		// Read in one full row of input feature map
		if(do_FW || do_dW){
            read_input(input_grad, input_storage, ROWS, b, WIDE_LEN);
		}

		// Read the gradient of the output layer as the input
		if(do_dX || do_dW){

			read_input(out_grad, output_grad_storage, COLS, b, WIDE_LEN);
		}

		// ZERO INITS
		// Zero initialize one full row of output feature map
		if(do_FW){
			zero_init_buffer(output_grad_storage);
		}

		if(do_dX){
			zero_init_buffer(result_storage);
		}

		// ================================================================================================ C
		for(unsigned c = 0; c < C_loop; c++){
	#pragma HLS LOOP_TRIPCOUNT min=1 max=8

			unsigned c_offset = c * TILE_SIZE;

			if(do_FW || do_dW){
                read_into_tile_buffer(A_, input_storage, c_offset);
			}

			if(do_dX || do_FW){
				read_weights<WIDE_LEN>(weight, weight_buffer, COLS, c_offset);

			}

			zero_init_tile_buffer(X2_);

			// ================================================================================================ F
			for(unsigned f = 0; f < F_loop; f++){
	#pragma HLS LOOP_TRIPCOUNT min=1 max=8
	#pragma HLS PIPELINE II=1
				unsigned f_start = f * TILE_SIZE;

				if(do_FW){
					read_into_tile_buffer(C_, output_grad_storage, f_start);
				}

				if(do_dX){
					read_into_tile_buffer(A_, output_grad_storage, f_start);
				}

				if(do_dW){
					read_into_tile_buffer(C_, output_grad_storage, f_start);
				}

				// ==== COMPUTATION CORE =============================================================
                for(unsigned ff = 0; ff < TILE_SIZE; ff++){
			    #pragma HLS UNROLL
					HLSNN_DataType tmp = C_[ff];

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
							last = X2_[ff][cc];
							B = weight_buffer[cc][ff][f];
						}

						if(do_dW){
							A_index = cc;
							B = tmp;
							last = 0.0;
						}

						HLSNN_DataType A = A_[A_index];
						HLSNN_DataType x_add = A * B + last;

						if(do_FW){
							tmp = x_add;
						}

						if(do_dX){
							X2_[ff][cc] = x_add;
						}

						if(do_dW){
							X2_[cc][ff] = x_add;
						}
					}

					if(do_FW){
						X_[ff] = tmp;
					}
				}

				// ===================================================================================

				if(do_FW){
					write_back_to_buffer(X_, output_grad_storage, f_start);
				}

				if(do_dW){
					write_back_to_weight_buffer(weight_buffer, X2_, f);
				}

			}

			if(do_dX){
				// Sum up values in X2_ along the C axes and store it in the large buffer
				reduce_and_store(X2_, result_storage, c_offset);
			}

			if(do_dW){
				read_sum_and_write_weights<WIDE_LEN>(weight, weight_buffer, weight_tmp_buffer, COLS, c_offset);
			}

			// ================================================================================================ F

		}
		// ================================================================================================ C

		if(do_FW){

			if(do_maxpool){
				HLSNN_DataType max_value = 0;

				HLSNN_DataType v1 = 0;
				HLSNN_DataType v2 = 0;

				unsigned off = b * F;

				unsigned pooling_index = 0;

				unsigned max_index = 0;
				bool flip = true;
				for(unsigned f = 0; f < F-1; f++){

					HLSNN_DataType tmp = output_grad_storage[f];
					HLSNN_DataType b_ = bias_storage[b];
					HLSNN_DataType v = tmp + b_;

					if(v < 0){
						v = 0;
					}

					if(flip){
						v1 = v;
						flip = !flip;
					}
					else{
						v2 = v;

						if(v1 >= v2){
							max_value = v1;
							max_index = 0;
						}
						else{
							max_value = v2;
							max_index = 1;
						}

						indices_buffer_maxpool[pooling_index] = max_index;
						output_grad_storage[pooling_index] = max_value;
						pooling_index++;
						flip = !flip;
					}

				}

				unsigned write_element = pooling_index + 1;
				unsigned write_start = (write_element * b);

				indices.write_request(write_start, write_element);
				for(unsigned ff = 0; ff < write_element; ff++){
			#pragma HLS PIPELINE II=1
					unsigned tmp = indices_buffer_maxpool[ff];
					indices.write(tmp);
				}
				indices.write_response();
			}

			unsigned cols = COLS;
			if(do_maxpool){
				cols = col_out;
			}

			unsigned K_loop = K;
			if(do_im2col){
				write_im2col(out_grad, output_grad_storage, bias_storage, cols, b, F_non_padded, K, WIDE_LEN, do_relu, !do_maxpool);
			}
			else if(do_col2im){
				write_col2im<WIDE_LEN>(out_grad, output_grad_storage, output_grad_storage_col2im, K_count, out_count, COLS, K);
			}
            else{
                write(out_grad, output_grad_storage, bias_storage, cols, F_non_padded, b, WIDE_LEN, do_relu, do_bias_end, do_fc);
            }

		}

		if(do_dX){
            write(input_grad, result_storage, bias_storage, ROWS, F_non_padded, b, WIDE_LEN, 0, 0, do_fc);
		}

	}

	// ================================================================================================ B

}
}
