#ifndef HLSNN_DENSE_HPP
#define HLSNN_DENSE_HPP

namespace hlsnn {

    template<class T, unsigned MAX, unsigned MaxM, unsigned MaxN>
    void Dense1D_array(T (&inM)[1][MAX],
    		T (&weight)[MaxM][MaxN],
			T (&bias)[1][MaxN],
			T (&outM)[1][MAX],
    		unsigned int M_kernel, unsigned int N_kernel)
    {
#pragma HLS INLINE off
    	for(unsigned n = 0; n < MaxN; n++){
    		T sum = 0;
    		for(unsigned m = 0; m < MaxM; m++){
#pragma HLS UNROLL factor=16
    			if(m < M_kernel){
    				T x = inM[0][m];
    				T w = weight[m][n];
    				T h = x * w;
    				sum += h;
    			}
    		}
            
    		if(n < N_kernel){
    			sum += bias[0][n];
    			outM[0][n] = sum;
    		}
    	}
    }

}

#endif
