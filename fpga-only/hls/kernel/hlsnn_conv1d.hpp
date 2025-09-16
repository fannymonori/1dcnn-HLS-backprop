#ifndef HLSNN_OPT_CONV_HPP
#define HLSNN_OPT_CONV_HPP

#include "hlsnn_matrix.hpp"
#include "hlsnn_tesnor4d.hpp"
#include "hlsnn_tensor3d.hpp"

namespace hlsnn {

    template<class T, unsigned MaxK, unsigned MaxCin,
			unsigned MaxCout, unsigned MaxM, unsigned MaxN>
    void OptNaiveConv1D_array(T (&matrixT)[MaxM][MaxN],
    		T (&kernelT)[MaxK][MaxCin][MaxCout],
            T (&biasInput)[1][MaxCout],
			T (&outT)[MaxM][MaxN],
			unsigned M_padded,
			unsigned kernel_size,
			unsigned cin,
			unsigned cout) {
#pragma HLS INLINE off
        int pad = kernel_size - 1;

        unsigned start = M_padded - pad;
        unsigned end = cin;
        OptNaiveConv1D_zero: for (int i = 0; i < MaxM; i++) {
            for (int j = 0; j < MaxCin; j++) {
            	if(i >= start && j < end){
            		matrixT[i][j] = 0;
            	}
            }
        }

        unsigned m_end = M_padded - kernel_size + 1;

        OptNaiveConv1D_comp: for (unsigned filter = 0; filter < MaxCout; filter++) {
			for (unsigned m = 0; m < MaxM; m++) {
#pragma HLS PIPELINE II=1
                T tmp = 0;
				for (unsigned channel = 0; channel < MaxCin; channel++) {
#pragma HLS UNROLL
                    T tmp_k = 0;
					for (unsigned k = 0; k < MaxK; k++) {
                    	if(k < kernel_size && filter < cout){
                    		T tmp_kernel = kernelT[k][channel][filter];
                    		T tmp_matrix = matrixT[m + k][channel];
                    		T mult = tmp_kernel * tmp_matrix;
                    		tmp_k += mult;
                    	}
                    }

                    if(channel < cin && filter < cout){
                    	tmp += tmp_k;
                    }
                }

				if(m < m_end && filter < cout){
					tmp += biasInput[0][filter];
					outT[m][filter] = tmp;
				}
            }
        }
    }

    template<class T, unsigned K, unsigned Cin, unsigned Cout, unsigned M, unsigned M_out>
    void OptNaiveConv1D(T (&matrixT)[M][Cin], T (&kernelT)[K][Cin][Cout], T (&outT)[M_out][Cout]) {
#pragma HLS INLINE off
        conv1dfw: for (unsigned filter = 0; filter < Cout; filter++) {
            for (unsigned m = 0; m < M - K + 1; m++) {
                T tmp = 0;
                for (unsigned channel = 0; channel < Cin; channel++) {
                    T tmp_k = 0;
                    for (unsigned k = 0; k < K; k++) {
                        tmp_k += kernelT[k][channel][filter] * matrixT[m + k][channel];
                    }
                    tmp += tmp_k;
                }
                outT[m][filter] = tmp;
            }
        }
    }

    template<class T, unsigned Cin, unsigned Cout, unsigned M_padded, unsigned M_out>
    void OptNaiveConv1D_bw_array(T (&in1)[M_padded][Cin],
    						T (&in2)[M_out][Cout],
							T (&outT)[M_padded - M_out + 1][Cin][Cout]) {
#pragma HLS INLINE off
        unsigned width = M_padded - M_out + 1;
        conv1dbw: for (unsigned k = 0; k < M_out; k++) {
            for (unsigned m = 0; m < width; m++) {
            	for (unsigned filter = 0; filter < Cout; filter++) {
#pragma HLS UNROLL factor=2
            		for (unsigned channel = 0; channel < Cin; channel++) {
                		outT[m][channel][filter] += in1[m + k][channel] * in2[k][filter];
                	}
            	}
        	}
        }
    }

}

#endif
