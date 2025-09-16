#ifndef HLSNN_ARITHMETIC_HPP
#define HLSNN_ARITHMETIC_HPP

namespace hlsnn {

	template<typename T, unsigned M, unsigned N>
	void sum_axis0_array(T (&m)[M][N], T (&result)[1][N]) {
#pragma HLS inline off

		for(unsigned j = 0; j < N; j++){
			result[0][j] = 0;
		}

		for(unsigned i = 0; i < M; i++){
			for(unsigned j = 0; j < N; j++){
				result[0][j] += m[i][j];
			}
		}
	}

	template<typename T, unsigned M1, unsigned N1, unsigned N2>
	void sum_axis0_array(T (&m)[M1][N1], T (&result)[1][N2], unsigned M, unsigned N) {
#pragma HLS inline off
		for (unsigned j = 0; j < N1; j++) {
			T sum = 0;
			for (unsigned i = 0; i < M1; i++) {
				if(i < M && j < N){
					sum += m[i][j];
				}
			}
			if(j < N){
				result[0][j] = sum;
			}
		}
	}

}

#endif //HLSNN_ARITHMETIC_HPP
