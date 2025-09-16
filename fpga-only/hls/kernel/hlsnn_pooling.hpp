#ifndef HLSNN_POOLING_HPP
#define HLSNN_POOLING_HPP

#include "hlsnn_matrix.hpp"
#include "hlsnn_tensor3d.hpp"
#include "hlsnn_tesnor4d.hpp"
#include "hlsnn_vector.hpp"

#include <iostream>

#define POOL_SIZE 2

namespace hlsnn
{
    
	template<class T, unsigned MAX_M, unsigned MAX_N>
	void MaxPool1D_array(T (&inM)[MAX_M][MAX_N], T (&outM)[MAX_M][MAX_N], unsigned M, unsigned N, unsigned S) {
	#pragma HLS INLINE off

		unsigned m_stop = M - POOL_SIZE + 1;

		maxpool2_ch: for(unsigned ch = 0; ch < MAX_N; ch++) {
			unsigned count = 0;
			maxpool2_m: for (unsigned m = 0; m < MAX_M; m = m + S) {
				if(m < m_stop && ch < N){
					T value1 = inM[m][ch];
					T value2 = inM[m + 1][ch];
					outM[count][ch] = value1 > value2 ? value1: value2;
					count++;
				}
			}
		}
	}

    template<unsigned S, class T, unsigned M, unsigned N, unsigned M2, unsigned N2>
    void MaxPool1D_backprop_array(T (&inX)[M][N],
                            T (&inGrad)[M2][N2],
							T (&outM)[M][N]) {
        maxpool1d_bw1_ch: for(unsigned ch = 0; ch < N; ch++) {
            unsigned count = 0;
            maxpool1d_bw1_m: for (unsigned m = 0; m < M - S + 1; m = m + S) {
                T value1 = inX[m][ch];
                T value2 = inX[m + 1][ch];
                int index_max = value1 >= value2 ? m : (m + 1);
                outM[index_max][ch] = inGrad[count][ch];
                count++;
            }
        }
    }

    template<unsigned S, class T, unsigned M1, unsigned N1, unsigned M2, unsigned N2, unsigned M3, unsigned N3>
    void MaxPool1D_backprop_array(
    		T (&inX)[M1][N1],
			T (&inGrad)[M2][N2],
			T (&outM)[M3][N3], unsigned M, unsigned N) {
#pragma HLS INLINE off

    	unsigned m_end = M - S + 1;
        maxpool1d_bw2_ch: for(unsigned ch = 0; ch < N1; ch++) {
            unsigned count = 0;
            maxpool1d_bw2_m: for (unsigned m = 0; m < M1; m = m + S) {
            	if(ch < N && m < m_end){
					T value1 = inX[m][ch];
					T value2 = inX[m + 1][ch];
					int index_max = value1 >= value2 ? m : (m + 1);
					outM[index_max][ch] = inGrad[count][ch];

					count++;
            	}
            }
        }
    }

}

#endif
