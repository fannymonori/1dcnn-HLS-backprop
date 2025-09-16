#ifndef HLSNN_MATRIX_OPS_HPP
#define HLSNN_MATRIX_OPS_HPP

#include <iostream>

namespace hlsnn {

	template<class T, unsigned M, unsigned N>
	void matrix_zero_init(T (&matrix)[M][N]) {
		matrix_zero_init_i: for (unsigned i = 0; i < M; i++) {
			matrix_zero_init_j: for (unsigned j = 0; j < N; j++) {
				matrix[i][j] = T(0);
			}
		}
	}


	template<class T, unsigned M, unsigned N>
	void matrix_init_from_array(T (&matrix)[M][N], T *t) {
		matrix_init_w_data1_i: for (unsigned i = 0; i < M; i++) {
			matrix_init_w_data1_j: for (unsigned j = 0; j < N; j++) {
				matrix[i][j] = t[j + i * (N)];
			}
		}
	}

	template<class T, unsigned M, unsigned N>
	void matrix_init_from_array(T (&matrix)[M][N], T *t, unsigned M_, unsigned N_) {
#pragma HLS INLINE off
		matrix_init_w_data2: for (unsigned i = 0; i < M; i++) {
			for (unsigned j = 0; j < N; j++) {
				if(i < M_ && j < N_){
					matrix[i][j] = t[j + i * (N)];
				}
			}
		}
	}


	template<class T, unsigned M, unsigned N>
	void matrix_copy(T (&matrix)[M][N], T (&matrix2)[M][N]) {
		matrix_copy_i: for (unsigned i = 0; i < M; i++) {
			matrix_copy_j: for (unsigned j = 0; j < N; j++) {
				matrix[i][j] = matrix2[i][j];
			}
		}
	}


	template<class T, unsigned M, unsigned N>
	void matrix_load_from_array(T (&matrix)[M][N], T *t, unsigned M_, unsigned N_) {
#pragma HLS INLINE off
		matrix_load1_i: for (unsigned i = 0; i < M; i++) {
			matrix_load1_j: for (unsigned j = 0; j < N; j++) {
				if(i < M_ && j < N_){
					matrix[i][j] = t[j + i * (N)];
				}
			}
		}
	}


	template<class T, unsigned M, unsigned N, unsigned S>
	void matrix_load_from_array(T (&matrix)[M][N], T (&matrix2)[S], unsigned M_, unsigned N_) {
#pragma HLS INLINE off
		matrix_load2_i: for (unsigned i = 0; i < M; i++) {
			matrix_load2_j: for (unsigned j = 0; j < N; j++) {
				if(i < M_ && j < N_){
					matrix[i][j] = matrix2[j + i * (N)];
				}
			}
		}
	}

	template<class T, unsigned M, unsigned N, unsigned M2, unsigned N2>
	void matrix_load_from_array(T (&matrix)[M][N], T (&matrix2)[M2][N2], unsigned M_, unsigned N_) {
#pragma HLS INLINE off
		matrix_load3_i: for (unsigned i = 0; i < M; i++) {
			matrix_load3_j: for (unsigned j = 0; j < N; j++) {
				if(i < M_ && j < N_){
					matrix[i][j] = matrix2[i][j];
				}
			}
		}
	}


	template<class T, unsigned M, unsigned N, unsigned M2, unsigned N2>
	void matrix_copy_to_array(T (&matrix)[M][N], T (&matrix2)[M2][N2], unsigned M_, unsigned N_) {
#pragma HLS INLINE off
		matrix_copy_to_array1_i: for (unsigned i = 0; i < M; i++) {
			matrix_copy_to_array1_j: for (unsigned j = 0; j < N; j++) {
				if(i < M_ && j < N_){
					matrix2[i][j] = matrix[i][j];
				}
			}
		}
	}

	template<class T, class T2, unsigned M, unsigned N>
	void matrix_copy_to_array(T (&matrix)[M][N], T2 (&matrix2)[M * N]) {
#pragma HLS INLINE off
		matrix_copy_to_array2_i: for (unsigned i = 0; i < M; i++) {
			matrix_copy_to_array2_j: for (unsigned j = 0; j < N; j++) {
				matrix2[j + i * (N)] = T2(matrix[i][j]);
			}
		}
	}

	template<class T, class T2, unsigned M, unsigned N>
	void matrix_copy_to_array(T (&matrix)[M][N], T2 (&matrix2)[M * N], unsigned M_, unsigned N_) {
#pragma HLS INLINE off
		matrix_copy_to_array3_i: for (unsigned i = 0; i < M; i++) {
			matrix_copy_to_array3_j: for (unsigned j = 0; j < N; j++) {
				if(i < M_ && j < N_){
					matrix2[j + i * (N)] = T2(matrix[i][j]);
				}
			}
		}
	}

	template<class T, class T2, unsigned M, unsigned N>
	void matrix_fill(T (&matrix)[M][N], T2 v) {
#pragma HLS INLINE off
        matrix_fill_w_v_m: for(int m = 0; m < M; m++){
        	matrix_fill_w_n: for(int n = 0; n < N; n++){
                matrix[m][n] = T(v);
            }
        }
	}

	template<class T, unsigned M, unsigned N>
	T matrix_sum(T (&matrix)[M][N]) {
#pragma HLS INLINE off
        T tmp = 0;
        matrix_sum_m: for(int m = 0; m < M; m++){
        	matrix_sum_n: for(int n = 0; n < N; n++){
                tmp += matrix[m][n] * matrix[m][n];
            }
        }

        return tmp;
	}

	template<class T, unsigned M, unsigned N>
	T matrix_avg(T (&matrix)[M][N]) {
#pragma HLS INLINE off
        T avg = hlsnn::matrix_sum(matrix);
        avg = avg / (M*N);
        return avg;
	}

	template<class T, unsigned M, unsigned N>
	T matrix_max(T (&matrix)[M][N]) {
#pragma HLS INLINE off
        T maxval = matrix[0][0];
        matrix_max_i: for (int i = 0; i < M; i++) {
        	matrix_max_j: for (int j = 0; j < N; j++) {
                if (matrix[i][j] > maxval)
                    maxval = matrix[i][j];
            }
        }

        return maxval;
	}

	template<class T, unsigned M, unsigned N>
	T min(T (&matrix)[M][N]) {
#pragma HLS INLINE off
        T minval = matrix[0][0];
        matrix_min_i: for (int i = 0; i < M; i++) {
        	matrix_min_j: for (int j = 0; j < N; j++) {
                if (matrix[i][j] > minval)
                    minval = matrix[i][j];
            }
        }
        return minval;
	}

	template<class T, unsigned M, unsigned N>
    void print(T (&matrix)[M][N]) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

	template<class T, unsigned M, unsigned N>
    void printSize(T (&matrix)[M][N]) {
        std::cout << "Size: (" << M << "," << N << ")" << std::endl;
    }

    template<typename T, unsigned M, unsigned N>
    void scalarMultiply(T (&matrix)[M][N], T b, T (&result)[M][N]) {
        scalar_multiply1_i: for (int i = 0; i < M; i++) {
        	scalar_multiply1_j: for (int j = 0; j < N; j++) {
                result[i][j] = matrix[i][j] * b;
            }
        }
    }

    template<typename T, unsigned M, unsigned N>
    void scalarMultiply(T (&matrix)[M][N], T b) {
    	scalar_multiply2_i: for (int i = 0; i < M; i++) {
    		scalar_multiply2_j: for (int j = 0; j < N; j++) {
            	matrix[i][j] = matrix[i][j] * b;
            }
        }
    }

    template<typename T, unsigned M, unsigned N, unsigned K, unsigned L>
    void matrixMultiply(T (&matrix1)[M][N], T (&matrix2)[K][L], T (&result)[M][L]) {
        matrix_multiply_i: for (int i = 0; i < M; i++) {
        	matrix_multiply_j: for (int j = 0; j < L; j++) {
            	T tmp = 0;
            	matrix_multiply_k: for (int k = 0; k < K; k++) {
                	tmp += matrix1[i][k] * matrix2[k][j];
                }

                result[i][j] = tmp;
            }
        }
    }

    template<typename T, unsigned M, unsigned N>
    void matrixAdd(T (&matrix1)[M][N], T (&matrix2)[M][N], T (&result)[M][N]) {
        matrix_add_i: for (unsigned i = 0; i < M; i++) {
        	matrix_add_j: for (unsigned j = 0; j < N; j++) {
            	result[i][j] = matrix1[i][j] + matrix2[i][j];
            }
        }
    }

    template<typename T, typename T2, unsigned M, unsigned N>
    void matrixSubtract(T (&matrix1)[M][N], T2 (&matrix2)[M][N], T (&result)[M][N]) {
        matrix_subtract1_i: for (unsigned i = 0; i < M; i++) {
        	matrix_subtract1_j: for (unsigned j = 0; j < N; j++) {
            	result[i][j] = matrix1[i][j] - T(matrix2[i][j]);
            }
        }
    }

    template<typename T, typename T2, unsigned M, unsigned N>
    void matrixSubtract(T (&matrix1)[M][N], T2 (&matrix2)[M][N]) {
    	matrix_subtract2_i: for (unsigned i = 0; i < M; i++) {
    		matrix_subtract2_j: for (unsigned j = 0; j < N; j++) {
            	matrix1[i][j] = matrix1[i][j] - T(matrix2[i][j]);
            }
        }
    }

    template<typename T, typename T2, unsigned M, unsigned N>
    void matrixSubtract2(T (&matrix1)[M][N], T2 (&matrix2)[M][N], bool &finished) {
    	finished = false;
    	matrix_subtract2_i: for (unsigned i = 0; i < M; i++) {
    		matrix_subtract2_j: for (unsigned j = 0; j < N; j++) {
            	matrix1[i][j] = matrix1[i][j] - T(matrix2[i][j]);

            	if(i == M && j == N){
            		finished = true;
            	}
            }
        }
    }

    template<typename T, unsigned M1, unsigned N1, unsigned M2, unsigned N2, unsigned MaxM, unsigned MaxN>
    void matrixMultiplyBounded_array(T (&a)[M1][N1],
    		T (&b)[M2][N2],
			T (&result)[MaxM][MaxN],
			unsigned M, unsigned K, unsigned L){
#pragma HLS inline off
    	matrixMultiplyBounded_array_i: for (int i = 0; i < M1; i++) {
    		matrixMultiplyBounded_array_j: for (int j = 0; j < N2; j++) {
            	T tmp = 0;
            	matrixMultiplyBounded_array_k: for (int k = 0; k < N1; k++) {
                	if(k < K && i < M && j < L){
						T v1 = a[i][k];
						T v2 = b[k][j];
						T v3 = v1 * v2;
						tmp += v3;
                	}
                }

                result[i][j] = tmp;
            }
        }
    }

    template<typename T, unsigned M, unsigned N>
    void matrix_sgd(T (&matrix1)[M][N], T (&matrix2)[M][N], T s, bool &flag) {
        matrix_sgd_i: for (unsigned i = 0; i < M; i++) {
            matrix_sgd_j: for (unsigned j = 0; j < N; j++) {
#pragma HLS UNROLL factor=2
                T v1 = matrix1[i][j];
                T v2 = matrix2[i][j];
                T lr = s;
                T mult = lr * v2;
                matrix1[i][j] = v1 - mult;

                if((i == (M-1)) && (j == (N-1))){
                    flag = true;
                }
            }
        }
    }


}

#endif
