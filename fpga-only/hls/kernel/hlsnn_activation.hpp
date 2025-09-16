#ifndef HLSNN_ACTIVATION_HPP
#define HLSNN_ACTIVATION_HPP

#include "hls_math.h"

namespace hlsnn
{

	enum activation_type { relu, sigmoid, tanh };

    template<class T, unsigned MAXM, unsigned MAXN>
    static void softmax_array(T (&inM)[MAXM][MAXN], T (&outM)[MAXM][MAXN], unsigned M, unsigned N){
        T expsum = T(0);
        softmax_expsum_i: for(int i = 0; i < MAXM; i++){
        	softmax_expsum_j: for(int j = 0; j < MAXN; j++){
            	if(i < M && j < N){
            		//expsum += hls::exp((ap_fixed<64, 32>)inM[i][j]);
            		expsum += hls::exp(inM[i][j]);
            		//expsum += std::exp(inM[i][j]);
            	}
            }
        }

        softmax_i: for(int i = 0; i < MAXM; i++){
        	softmax_j: for(int j = 0; j < MAXN; j++){
            	if(i < M && j < N){
            		//outM[i][j] = hls::exp((ap_fixed<64, 32>)inM[i][j]) / expsum;
            		outM[i][j] = hls::exp(inM[i][j]) / expsum;
            		//outM[i][j] = std::exp(inM[i][j]) / expsum;
            	}
            }
        }

    }

    template<class T, unsigned MaxM, unsigned MaxN>
    void ReLU_array(T (&inM)[MaxM][MaxN], T (&outM)[MaxM][MaxN], unsigned M, unsigned N) {
#pragma HLS INLINE off
        relu_i: for(unsigned i = 0; i < MaxM; i++) {
            relu_j: for (unsigned j = 0; j < MaxN; j++) {

                if(i < M && j < N){
                	T v = inM[i][j];
                    T tmp = v > T(0) ? v : T(0);
                    outM[i][j] = tmp;
                }
            }
        }
    }

    template<class T, unsigned M, unsigned N>
    void ReLU_backprop_array(T (&inX)[M][N], T (&inGrad)[M][N], T (&outM)[M][N]) {
        relubw_i: for(unsigned i = 0; i < M; i++) {
            relubw_j: for (unsigned j = 0; j < N; j++) {
                T v = inX[i][j];
                T v2 = inGrad[i][j];
                T tmp = v > T(0) ? v2 : T(0);
                outM[i][j] = tmp;
            }
        }
    }

    template<class T, unsigned M, unsigned N, unsigned M2, unsigned M3>
    void ReLU_backprop_paddedIn_array(T (&inX)[M2][N],
        	T (&inGrad)[M][N],
			T (&outM)[M3][N]) {
            relu_bw_padded_i: for(unsigned i = 0; i < M; i++) {
            	relu_bw_padded_j: for (unsigned j = 0; j < N; j++) {
                T v = inX[i][j];
                T tmp = v > T(0) ? inGrad[i][j] : T(0);
                outM[i][j] = tmp;
            }
        }
    }

}

#endif
