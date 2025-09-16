#ifndef HLSNN_REORDER_HPP
#define HLSNN_REORDER_HPP

namespace hlsnn {

    template<class T, unsigned M, unsigned N, unsigned C>
    void Flip_axis1_array(T (&inM)[M][N][C], T (&outM)[M][N][C])
    {
        int count_m = M -1;
        flip_axis1_i: for(int i = 0; i < M; i++)
        {
        	flip_axis1_j: for(int j = 0; j < N; j++)
            {
        		flip_axis1_c: for(int c = 0; c < C; c++) {
                    outM[count_m][j][c] = inM[i][j][c];
                }
            }
            count_m--;
        }
    }

}

#endif //HLSNN_REORDER_HPP
