#include "types.hpp"

void transpose_matrix_widened(wide_type *in, wide_type *out, unsigned M, unsigned N, unsigned wide_length){

	unsigned m_widened = M / wide_length < 1 ? 1 : M / wide_length;
	unsigned n_widened = N / wide_length < 1 ? 1 : N / wide_length;

	unsigned index = 0;
	for(int i = 0; i < M; i++){
		unsigned col_index = 0;
		for(int j = 0; j < n_widened; j++){
			wide_type tmp;
			for(int k = 0; k < wide_length; k++){
				unsigned row = col_index * m_widened ;
				out[(i / wide_length) + (col_index * m_widened)][i % wide_length] = in[index][k];
				col_index++;
			}
			index++;
		}
	}
}