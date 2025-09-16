#include <vector>
#include <zlib.h>
#include "cnpy.h"
#include "types.hpp"

// ====================================================================================================================
// READ PYTHON NPZ FILE

template <typename T>
void readNpz(std::string path, std::vector<std::string>& layer_names, std::map<std::string, std::vector<T>>& npz_map){

    cnpy::npz_t data_npz = cnpy::npz_load(path);

    for(unsigned i = 0; i < layer_names.size(); i++) {

        std::string layer_name = layer_names.at(i);
        cnpy::NpyArray layer = data_npz[layer_name];

        std::vector<T> vec1;
        if(layer.data_holder)
            vec1 = layer.as_vec<T>();

        npz_map.emplace(layer_name, vec1);
    }

}

// ====================================================================================================================
// PRINTING FUNCTIONS

/**
 Print matrix in a flattened way.
*/
void print_flattened_matrix(wide_type* matrix, int M_widened, int M) {
    int count = 0;
    for (size_t i = 0; i < M_widened && count < M; i++) {
        for (size_t j = 0; j < WIDE_LEN && count < M; j++) {
            std::cout << matrix[i][j] << " ";
            count++;
        }
    }
    std::cout << std::endl;
}

/**
 Print matrix in a flattened way.
*/
void print_flattened_matrix(std::vector<wide_type>& matrix, int M, std::ofstream &out_file) {
    int count = 0;
    for (size_t i = 0; i < matrix.size() && count < M; i++) {
        for (size_t j = 0; j < WIDE_LEN && count < M; j++) {
            out_file << matrix[i][j] << " ";
            count++;
        }
    }
    out_file << std::endl;
}

/**
 Print matrix in a flattened way.
*/
void print_flattened_matrix(wide_type* matrix, int M_widened, int M, std::ofstream &out_file) {
    int count = 0;
    for (size_t i = 0; i < M_widened && count < M; i++) {
        for (size_t j = 0; j < WIDE_LEN && count < M; j++) {
            out_file << matrix[i][j] << " ";
            count++;
        }
    }
    out_file << std::endl;
}

/**
 Print matrix to a file.
*/
template<class T>
void print_output(T *array, unsigned M, unsigned N, std::ofstream &out_file){
	unsigned c = 0;
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
            out_file << array[c] << " ";
			c++;
		}
        out_file << std::endl;
	}
    out_file << "==============" << std::endl;
	out_file << std::endl;
}

/**
 Print matrix to a file.
*/
template<class T>
void print_output(T *array, unsigned M, unsigned N){
	unsigned c = 0;
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N; j++){
            std::cout << array[c] << " ";
			c++;
		}
        std::cout << std::endl;
	}
    std::cout << "==============" << std::endl;
	std::cout << std::endl;
}

/**
 Print vector of fixed point type.
*/
void print_vector(HLSNN_DataType *vec, unsigned length){
    for (int i = 0; i < length; i++) {
    	std::cout << vec[i] << " ";
    }
    std::cout << std::endl;

}

/**
 Print vector of type unsigned.
*/
void print_vector(unsigned *vec, unsigned length){
    for (int i = 0; i < length; i++) {
    	std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

/**
 Print vector of fixed point type to file.
*/
void print_vector(HLSNN_DataType *vec, unsigned length, std::ofstream &out_file){
    for (int i = 0; i < length; i++) {
    	out_file << vec[i] << " ";
    }
    out_file << std::endl;

}

/**
 Print vector of type unsigned to file.
*/
void print_vector(unsigned *vec, unsigned length, std::ofstream &out_file){
    for (int i = 0; i < length; i++) {
    	out_file << vec[i] << " ";
    }
    out_file << std::endl;
}

/**
 Print matrix stored in a continuous and tiled way to file.
*/
void print_output_wide(wide_type *array, unsigned M, unsigned N, unsigned wide_len, std::ofstream &out_file){
	unsigned c = 0;

	unsigned loop_end = N / wide_len < 1 ? 1 : N / wide_len;

    if(N % wide_len != 0){
		loop_end++;
	}

	for(int i = 0; i < M; i++){
		for(int j = 0; j < loop_end; j++){

			wide_type tmp = array[c];
			for(int k = 0; k < wide_len; k++){
                out_file << tmp[k] << " ";
			}
			c++;
		}
        out_file << std::endl;
	}

    out_file << "==============" << std::endl;
    out_file << std::endl;
}

/**
 Print matrix stored in a continuous and tiled way (pointer array).
*/
void print_output_wide(wide_type *array, unsigned M, unsigned N, unsigned wide_len){
	unsigned c = 0;

	unsigned loop_end = N / wide_len < 1 ? 1 : N / wide_len;

    if(N % wide_len != 0){
		loop_end++;
	}

	for(int i = 0; i < M; i++){
		for(int j = 0; j < loop_end; j++){

			wide_type tmp = array[c];
			for(int k = 0; k < wide_len; k++){
                std::cout << tmp[k] << " ";
			}
			c++;
		}
        std::cout << std::endl;
	}
    std::cout << "==============" << std::endl;
    std::cout << std::endl;
}


// ====================================================================================================================
// COPYING FUNCTIONS

/**
 Read and copy data from continuous pointer array to a wide (tiled) format.
*/
void place_in_wide(HLSNN_DataType *ptr, wide_type *wide_vector, unsigned m, unsigned n, unsigned wide_length){

	unsigned widened_length = n/wide_length;
	unsigned diff = 0;

	if(n % wide_length != 0){
		widened_length = widened_length + 1;
		diff = n % wide_length;
	}

	unsigned index = 0;
    unsigned cont_index = 0;
	for(int i = 0; i < m; i++){
		for(int j = 0; j < widened_length; j++){
			wide_type tmp;
			for(int k = 0; k < wide_length; k++){
				HLSNN_DataType v = ptr[index];
				tmp[k] = v;
				index++;
			}
            wide_vector[cont_index] = tmp;
            cont_index++;
		}
	}
}

/**
 Copy from pointer vector to pointer vector.
*/
void place_in_vector(HLSNN_DataType *vec, HLSNN_DataType *ptr, unsigned length){
    for (int i = 0; i < length; i++) {
    	vec[i] = ptr[i];
    }
}

void place_in_vector_pad(HLSNN_DataType *vec, HLSNN_DataType *ptr, unsigned length, unsigned n){
    for (int i = 0; i < length; i++) {
    	if(i < n){
            vec[i] = ptr[i];
    	}
    	else{
            vec[i] = HLSNN_DataType(0.0);
    	}
    }
}

void place_in_wide_widen(HLSNN_DataType *ptr, wide_type *wide_vector, unsigned m, unsigned n, unsigned n_widened, unsigned wide_length){

	unsigned widened_length = n_widened/wide_length;

	unsigned col_count = 0;
	unsigned index = 0;
    unsigned cont_index = 0;
	for(int i = 0; i < m; i++){
		col_count = 0;
		for(int j = 0; j < widened_length; j++){
			wide_type tmp;
			for(int k = 0; k < wide_length; k++){
				HLSNN_DataType v = HLSNN_DataType(0.0);
				if(col_count < n){
					v = ptr[index];
					index++;
				}
				tmp[k] = v;
				col_count++;
			}

            wide_vector[cont_index] = tmp;
            cont_index++;
		}
	}

}

// ====================================================================================================================
// OTHER FUNCTIONS

float gen_random(){
	float MIN_RAND = -1.0, MAX_RAND = 1.0;
	const float range = MAX_RAND - MIN_RAND;
	float r =  range * (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) + MIN_RAND;
	return r;
}
