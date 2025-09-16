template<class T, unsigned K, unsigned SIGNAL_LENGTH,
			unsigned C, unsigned F>
void CONV1D_ARRAY(
    T (&IFM)[SIGNAL_LENGTH][C],
    T (&W)[K][C][F],
    T (&B)[F],
	T (&OFM)[SIGNAL_LENGTH][F],
	unsigned signal_length,
    unsigned padding,
	unsigned kernel_size,
	unsigned cin,
	unsigned cout
    ) {
    int pad = kernel_size - 1;

    unsigned m_end = (signal_length + padding) - kernel_size + 1;

    for (unsigned k = 0; k < kernel_size; k++) {
        for (unsigned m = 0; m < m_end; m++) {
            for (unsigned filter = 0; filter < cout; filter++) {
                T tmp = 0.0;
                for (unsigned ch = 0; ch < cin; ch++) {
                    T mult = W[k][ch][filter] * IFM[m + k][ch];
                    tmp += mult;
                }

			    tmp += B[filter];
			    OFM[m][filter] = tmp;
            }
        }
    }

}

template<class T, unsigned K, unsigned SIGNAL_LENGTH, unsigned C, unsigned F>
void CONV1D_ARRAY_DW(
    T (&IFM)[SIGNAL_LENGTH][C],
    T (&W_grad)[K][C][F],
	T (&OFM_grad)[SIGNAL_LENGTH][F],
	unsigned signal_length,
    unsigned padding,
	unsigned kernel_size,
	unsigned cin,
	unsigned cout
    ) {
    int pad = kernel_size - 1;

    unsigned m_end = (signal_length + padding) - kernel_size + 1;

    unsigned M_out = signal_length;
    unsigned M_padded = signal_length + padding;
    unsigned width = M_padded - M_out + 1;
    for (unsigned filter = 0; filter < cout; filter++) {
        for (unsigned m = 0; m < width; m++) {
            for (unsigned channel = 0; channel < cin; channel++) {
                T tmp = 0;
                for (unsigned k = 0; k < M_out; k++) {
                    tmp += IFM[m + k][channel] * OFM_grad[k][filter];
                }
                W_grad[m][channel][filter] += tmp;
            }
        }
    }
}


template<class T, unsigned F2>
void MM_CPU(
    T **inM,
    T **outM,
    T **W,
    T (&bias)[F2],
    unsigned b,
    unsigned c,
    unsigned f,
    bool do_bias
    ){

    if(do_bias){
        for(unsigned i = 0; i < b; i++){
            for(unsigned j = 0; j < f; j++){
                outM[i][j] = T(0.0);
                for(unsigned k = 0; k < c; k++){
                    outM[i][j] += inM[i][k] * W[k][j];
                }
                outM[i][j] += bias[j];
            }
        }
    }
    else{
        for(unsigned i = 0; i < b; i++){
            for(unsigned j = 0; j < f; j++){
                outM[i][j] = T(0.0);
                for(unsigned k = 0; k < c; k++){
                    outM[i][j] += inM[i][k] * W[k][j];
                }
            }
        }
    }
}

template<class T>
void MM_CPU_dW(
    T **inM,
    T **outM,
    T **W,
    unsigned b,
    unsigned c,
    unsigned f
    ){

    for(unsigned i = 0; i < b; i++){
        for(unsigned cc = 0; cc < c; cc++){
            for(unsigned ff = 0; ff < f; ff++){
                W[cc][ff] += inM[b][cc] * outM[b][ff];
            }            
        }
    }
}

template<class T, unsigned MAX_M, unsigned MAX_N>
void MaxPool1D(
    T (&inM)[MAX_M][MAX_N],
    T (&outM)[MAX_M][MAX_N],
    unsigned M,
    unsigned N,
    unsigned S) {

    unsigned pool_size = 2;
    unsigned m_stop = M - pool_size + 1;

    for(unsigned ch = 0; ch < N; ch++) {
        unsigned count = 0;
        for (unsigned m = 0; m < M; m = m + S) {
            if(m < m_stop && ch < N){
                T value1 = inM[m][ch];
                T value2 = inM[m + 1][ch];
				outM[count][ch] = value1 > value2 ? value1: value2;
				count++;
			}
		}
	}
}

template<class T, unsigned MAX_M, unsigned MAX_N>
void MaxPool1D_backprop(
    T (&inX)[MAX_M][MAX_N],
    T (&inGrad)[MAX_M][MAX_N],
    T (&outM)[MAX_M][MAX_N],
    unsigned M,
    unsigned N,
    unsigned S
    ) {
    for(unsigned ch = 0; ch < N; ch++) {
        unsigned count = 0;
        for (unsigned m = 0; m < M - S + 1; m = m + S) {
            T value1 = inX[m][ch];
            T value2 = inX[m + 1][ch];
            int index_max = value1 >= value2 ? m : (m + 1);
            outM[index_max][ch] = inGrad[count][ch];
                count++;
        }
    }
}
