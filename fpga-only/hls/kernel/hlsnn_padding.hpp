#ifndef NN_TRAIN_HLSNN_PADDING_HPP
#define NN_TRAIN_HLSNN_PADDING_HPP

namespace hlsnn {

    template<unsigned P, class T, unsigned M, unsigned N>
    void Padding_axis0_before_array(T (&inM)[M][N], T (&outM)[M+P][N])
    {
        padding_before1_i: for(unsigned i = 0; i < P; i++)
        {
        	padding_before1_j: for(unsigned j = 0; j < N; j++)
            {
                outM[i][j] = 0;
            }
        }

        padding_before2_i: for(unsigned i = P; i < M + P; i++)
        {
        	padding_before2_j: for(unsigned j = 0; j < N; j++)
            {
                outM[i][j] = inM[i - P][j];
            }
        }
    }

}

#endif //NN_TRAIN_HLSNN_PADDING_HPP
