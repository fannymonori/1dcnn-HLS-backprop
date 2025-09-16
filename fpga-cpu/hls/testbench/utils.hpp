#include <iostream>
#include <fstream>
#include "types.hpp"
#include "cnpy.h"

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

void print_output_wide(wide_type *array, unsigned M, unsigned N, unsigned wide_len, std::ofstream &out_file){
	unsigned c = 0;

	unsigned loop_end = N / wide_len < 1 ? 1 : N / wide_len;

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

void print_vector(std::vector<HLSNN_DataType> &vec, unsigned length){
    for (int i = 0; i < length; i++) {
    	std::cout << vec[i] << " ";
    }
    std::cout << std::endl;

}

void print_vector(std::vector<HLSNN_DataType> &vec, unsigned length, std::ofstream &out_file){
    for (int i = 0; i < length; i++) {
    	out_file << vec[i] << " ";
    }
    out_file << std::endl;

}

void print_vector(std::vector<unsigned> &vec, unsigned length){
    for (int i = 0; i < length; i++) {
    	std::cout << vec[i] << " ";
    }
    std::cout << std::endl;

}

void print_vector(std::vector<unsigned> &vec, unsigned length, std::ofstream &out_file){
    for (int i = 0; i < length; i++) {
    	out_file << vec[i] << " ";
    }
    out_file << std::endl;

}

void place_in_wide(HLSNN_DataType *ptr, std::vector<wide_type> &wide_vector, unsigned M, unsigned N, unsigned wide_length){

	unsigned widened_length = N/wide_length;
	unsigned diff = 0;

	if(N % wide_length != 0){
		widened_length = widened_length + 1;
		diff = N % wide_length;
	}

	unsigned index = 0;
	for(int i = 0; i < M; i++){
		for(int j = 0; j < widened_length; j++){
			wide_type tmp;
			for(int k = 0; k < wide_length; k++){
				HLSNN_DataType v = ptr[index];
				tmp[k] = v;
				index++;
			}
			wide_vector.push_back(tmp);
		}
	}
}

void place_in_wide_widen(HLSNN_DataType *ptr, std::vector<wide_type> &wide_vector, unsigned M, unsigned N, unsigned n_widened, unsigned wide_length){

	unsigned widened_length = n_widened/wide_length;

	unsigned col_count = 0;
	unsigned index = 0;
	for(int i = 0; i < M; i++){
		col_count = 0;
		for(int j = 0; j < widened_length; j++){
			wide_type tmp;
			for(int k = 0; k < wide_length; k++){
				HLSNN_DataType v = HLSNN_DataType(0.0);
				if(col_count < N){
					v = ptr[index];
					index++;
				}
				tmp[k] = v;
				col_count++;
			}
			wide_vector.push_back(tmp);
		}
	}
}

void place_in_vector(std::vector<HLSNN_DataType> &vec, HLSNN_DataType *ptr, unsigned length){

    for (int i = 0; i < length; i++) {
    	vec.push_back(ptr[i]);
    }

}

void place_in_vector_pad(std::vector<HLSNN_DataType> &vec, HLSNN_DataType *ptr, unsigned length, unsigned n){

    for (int i = 0; i < length; i++) {
    	if(i < n){
    		vec.push_back(ptr[i]);
    	}
    	else{
    		vec.push_back(HLSNN_DataType(0.0));
    	}
    }
}

template<unsigned SIZE>
void copy_into_array(HLSNN_DataType (&array)[SIZE], HLSNN_DataType *ptr, unsigned length){
    for (int i = 0; i < length; i++) {
    	array[i] = ptr[i];
    }
}

template<unsigned SIZE1, unsigned SIZE2>
void copy_into_array(HLSNN_DataType (&array)[SIZE1][SIZE2], HLSNN_DataType *ptr){
    unsigned l = 0;
    for(int i = 0; i < SIZE1; i++){
    	for(int j = 0; j < SIZE2; j++){
    		array[i][j] = ptr[l];
    		l++;
    	}
    }
}

void place_in_wide_pad(HLSNN_DataType *ptr, std::vector<wide_type> &wide_vector, unsigned M, unsigned N, unsigned wide_length){
	unsigned loop_bound = N > wide_length? N / wide_length : 1;
	unsigned widened_length = N/wide_length;

	unsigned index = 0;
	for(int i = 0; i < M; i++){
		for(int j = 0; j < loop_bound; j++){

			wide_type tmp;
			for(int k = 0; k < wide_length; k++){
				if(k < N){
					HLSNN_DataType v = ptr[index];
					tmp[k] = v;
					index++;
				}
				else{
					tmp[k] = 0.0;
				}
			}
			wide_vector.push_back(tmp);
		}
	}
}

void zero_pad_after_wide(wide_type *ptr, unsigned M, unsigned N, unsigned p, unsigned wide_length){
	unsigned widened_length = N / wide_length;

	if(N % wide_length != 0){
		widened_length = widened_length + 1;
	}

	unsigned index = 0;
	for(int i = 0; i < M; i++){
		unsigned col = 0;
		for(int j = 0; j < widened_length; j++){

			wide_type tmp = ptr[index];
			for(int k = 0; k < wide_length; k++){
				if(col > p){
					ptr[index][k] = 0.0;
				}
				col++;
			}

			index++;
		}
	}
}

template<unsigned SIZE>
void zero_init_array(HLSNN_DataType (&array)[SIZE]){
    for (int i = 0; i < SIZE; i++) {
    	array[i] = 0.0;
    }
}

void zero_init_vector(std::vector<HLSNN_DataType> &array, unsigned SIZE){
    for (int i = 0; i < SIZE; i++) {
    	array[i] = 0.0;
    }
}


void zero_init_wide(wide_type *array, unsigned M, unsigned N, unsigned wide_len){
	unsigned c = 0;
	unsigned loop_end = N / wide_len < 1 ? 1 : N / wide_len;

	for(int i = 0; i < M; i++){
		for(int j = 0; j < loop_end; j++){

			for(int k = 0; k < wide_len; k++){
				array[c][k] = 0.0;
			}
			c++;
		}
	}
}

