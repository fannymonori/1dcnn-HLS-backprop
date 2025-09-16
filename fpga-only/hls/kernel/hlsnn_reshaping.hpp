#ifndef HLSNN_RESHAPING_HPP
#define HLSNN_RESHAPING_HPP

namespace hlsnn
{
    
	template<typename T, unsigned MaxM, unsigned MaxN, unsigned MaxM2, unsigned MaxN2>
	void Transpose_array(T (&inM)[MaxM][MaxN], T (&outM)[MaxM2][MaxN2], unsigned M, unsigned N) {
#pragma HLS INLINE off
        transpose: for(int j = 0; j < MaxN; j++)
        {
            for(int i = 0; i < MaxM; i++)
            {
            	if(j < N && i < M){
            		outM[j][i] = inM[i][j];
            	}
            }
        }
	}

    template<class T, unsigned MaxM, unsigned MaxN, unsigned D>
    void Transpose_and_Flatten2D_array(T (&inM)[MaxM][MaxN], T (&outM)[1][D], unsigned M, unsigned N)
    {
#pragma HLS INLINE off
        int count = 0;
        transpose_and_flatten: for(int j = 0; j < MaxN; j++)
        {
            for(int i = 0; i < MaxM; i++)
            {
            	if(j < N && i < M){
            		outM[0][count] = inM[i][j];
                	count++;
            	}
            }
        }
    }

    template<class T, unsigned M, unsigned N>
    void Transpose_and_unFlatten2D_array(T (&inM)[1][M*N], T (&outM)[M][N])
    {
        int count = 0;
        transpose_and_unflatten: for(int j = 0; j < N; j++)
        {
            for(int i = 0; i < M; i++)
            {
                outM[i][j] = inM[0][count];
                count++;
            }
        }
    }

    template<class T, unsigned M, unsigned N, unsigned C>
    void SwapAxes_1_2_array(T (&inM)[M][N][C], T (&outM)[M][C][N])
    {
        swap_axes: for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                for(int c = 0; c < C; c++) {
                    outM[i][c][j] = inM[i][j][c];
                }
            }
        }
    }

}

#endif