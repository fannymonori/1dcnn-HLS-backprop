#ifndef HLSNN_TENSOR3D_OPS_HPP
#define HLSNN_TENSOR3D_OPS_HPP

namespace hlsnn {

	template<class T, unsigned M, unsigned N, unsigned C, unsigned S>
	void tensor3d_load_data(T (&tensor)[M][N][C], T (&t)[S], unsigned M_, unsigned N_, unsigned C_) {
#pragma HLS INLINE off
		tensor_load1_m: for (unsigned m = 0; m < M; m++) {
			tensor_load1_n: for (unsigned n = 0; n < N; n++) {
				tensor_load1_c: for (unsigned c = 0; c < C; c++) {
    	    		if(m < M_ && n < N_ && c < C_){
    	    			tensor[m][n][c] = t[c + n * C_ + m * (N_ * C_)];
    	    		}
    	        }
    		}
    	}
	}

	template<class T, unsigned M, unsigned N, unsigned C>
	void tensor3d_load_data(T (&tensor)[M][N][C], T (&tensor2)[M][N][C]) {
        tensor_load2_i: for (unsigned i = 0; i < M; i++) {
        	tensor_load2_j: for (unsigned j = 0; j < N; j++) {
        		tensor_load2_c: for (unsigned c = 0; c < C; c++) {
                	tensor[i][j][c] = tensor2[i][j][c];
                }
            }
        }
	}

	template<class T, unsigned M, unsigned N, unsigned C, unsigned M2, unsigned N2, unsigned C2>
	void tensor3d_load_data(T (&tensor)[M][N][C], T (&tensor2)[M2][N2][C2], unsigned M_, unsigned N_, unsigned C_) {
		tensor_load3_i: for (unsigned i = 0; i < M; i++) {
			tensor_load3_j: for (unsigned j = 0; j < N; j++) {
				tensor_load3_c: for (unsigned c = 0; c < C; c++) {
                	if(i < M_ && j < N_ && c < C_){
                		tensor[i][j][c] = tensor2[i][j][c];
                	}
                }
            }
        }
	}

	template<class T, class T2, unsigned M, unsigned N, unsigned C>
	void tensor3d_fill(T (&tensor)[M][N][C], T2 v) {
        tensor_fill_m: for (unsigned m = 0; m < M; m++) {
        	tensor_fill_n: for (unsigned n = 0; n < N; n++) {
        		tensor_fill_c: for (unsigned c = 0; c < C; c++) {
                	tensor[m][n][c] = T2(v);
                }
            }
        }
	}

    template<class T, unsigned M, unsigned N, unsigned L>
    void tensor_sgd(T (&m1)[M][N][L], T (&m2)[M][N][L], T s1, bool &flag) {
        flag = false;
    	for (unsigned i = 0; i < M; i++) {
    		for (unsigned j = 0; j < N; j++) {
    			for (unsigned l = 0; l < L; l++) {
                    T v_orig = m1[i][j][l];
                    T v_new = m2[i][j][l];
                    T lr = s1;
                    T mult = v_new * lr;
                    T subtr = v_orig - mult;
                    
                    m1[i][j][l] = subtr;

                    if((i == (M-1)) && (j == (N-1)) && (l == (L-1))){
                        flag = true;
                    }
    			}
    		}
    	}
    }
}


#endif