template<typename T, unsigned K, unsigned SIGNAL_LENGTH,
			unsigned C, unsigned F>
void fill_conv_arrays_random(
    T (&IFM)[SIGNAL_LENGTH][C],
    T (&W)[K][C][F],
    T (&W_flip)[K][F][C],
    T (&B)[F],
	T (&OFM)[SIGNAL_LENGTH][F],
    T (&OFM2)[SIGNAL_LENGTH][F],
    T (&OFM3)[SIGNAL_LENGTH][F],
    unsigned signal_length,
    unsigned padding,
    unsigned kernel_size,
    unsigned cin,
    unsigned cout
){

    for(unsigned k = 0; k < kernel_size; k++){
        for(unsigned c = 0; c < cin; c++){
            for(unsigned f = 0; f < cout; f++){
                W[k][c][f] = gen_random();
                W_flip[k][f][c] = gen_random();
            }
        }
    }

    for(unsigned m = 0; m < signal_length + padding; m++){
        for(unsigned c = 0; c < cin; c++){
            IFM[m][c] = gen_random();
        }
    }

    for(unsigned m = 0; m < signal_length + padding; m++){
        for(unsigned f = 0; f < cout; f++){
            OFM[m][f] = gen_random();
            OFM2[m][f] = gen_random();
            OFM3[m][f] = gen_random();
        }
    }
    
    for(unsigned f = 0; f < cout; f++){
        B[f] = gen_random();
    }

}


template<typename T>
void fill_conv_arrays_random(
    T *IFM,
    T* W,
    T *W_flip,
    T *B,
    T *OFM,
    T *OFM2,
    T *OFM3,
    unsigned SIGNAL_LENGTH,
    unsigned padding,
    unsigned K,
    unsigned C,
    unsigned F
){

    for(unsigned k = 0; k < K; k++){
        for(unsigned c = 0; c < C; c++){
            for(unsigned f = 0; f < F; f++){
                W[k][c][f] = gen_random();
                W_flip[k][f][c] = gen_random();
            }
        }
    }

    for(unsigned m = 0; m < SIGNAL_LENGTH + padding; m++){
        for(unsigned c = 0; c < C; c++){
            IFM[m][c] = gen_random();
        }
    }

    for(unsigned m = 0; m < SIGNAL_LENGTH + padding; m++){
        for(unsigned f = 0; f < F; f++){
            OFM[m][f] = gen_random();
            OFM2[m][f] = gen_random();
            OFM3[m][f] = gen_random();
        }
    }
    
    for(unsigned f = 0; f < F; f++){
        B[f] = gen_random();
    }

}

