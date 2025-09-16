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

void read_data_from_npz(std::string &path,
            std::vector<std::string> &layer_names,
            std::map<std::string, std::vector<HLSNN_DataType>> &dataMap,
            std::map<std::string, std::vector<double>> &dataMap_float
            ){

    readNpz(path, layer_names, dataMap_float);    

    std::map<std::string, std::vector<double>>::iterator it;
    for (it = dataMap_float.begin(); it != dataMap_float.end(); it++)
    {
    	std::vector<double> tmp = it->second;
    	std::vector<HLSNN_DataType> tmp_result;

        for(int i = 0; i < tmp.size(); i++){
        	if(std::isnan(tmp[i])){
        		std::cout << "nan" << std::endl;
        		tmp_result.push_back(HLSNN_DataType(0.0));
        	}
        	else{
        		tmp_result.push_back(HLSNN_DataType(tmp[i]));
        	}
        }

        dataMap.insert({it->first, tmp_result});

    }
}

// ====================================================================================================================
// PRINTING FUNCTIONS

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


void print_output_vector_wide(std::vector<wide_type> &array, unsigned M, unsigned N, unsigned wide_len){
	unsigned c = 0;

	unsigned loop_end = N / wide_len < 1 ? 1 : N / wide_len;

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

void print_output_vector_wide(std::vector<wide_type> &array, unsigned M, unsigned N, unsigned wide_len, std::ofstream &out_file){
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


// ====================================================================================================================
// COPYING HELPER FUNCTIONS

void place_in_wide(HLSNN_DataType *ptr, std::vector<wide_type> &wide_vector, unsigned m, unsigned n, unsigned wide_length){
	unsigned widened_length = n/wide_length;

	unsigned index = 0;
	for(int i = 0; i < m; i++){
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

void place_in_vector(std::vector<HLSNN_DataType> &vec, HLSNN_DataType *ptr, unsigned length){

    for (int i = 0; i < length; i++) {
    	vec[i] = ptr[i];
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

void place_in_wide_pad(HLSNN_DataType *ptr, std::vector<wide_type> &wide_vector, unsigned m, unsigned n, unsigned wide_length){
	unsigned loop_bound = n > wide_length? n / wide_length : 1;
	unsigned widened_length = n/wide_length;

	unsigned index = 0;
	for(int i = 0; i < m; i++){
		for(int j = 0; j < loop_bound; j++){

			wide_type tmp;
			for(int k = 0; k < wide_length; k++){
				if(k < n){
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


// ====================================================================================================================
// ZERO INIT FUNCTIONS

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

// ====================================================================================================================
// BUFFER LOADING

template<class T>
void load_into_buffer(T* buffer_ptr, T* storage_ptr,unsigned length){
    for(int j = 0; j < length; j++){
		buffer_ptr[j] = storage_ptr[j];
	}
}

// ====================================================================================================================
// GENERATING RANDOM ARRAYS

float gen_random(){
	float MIN_RAND = -1.0, MAX_RAND = 1.0;
	const float range = MAX_RAND - MIN_RAND;
	float r =  range * (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) + MIN_RAND;
	return r;
}

void fill_random(HLSNN_DataType* array, unsigned length){
    for(unsigned i = 0; i < length; i++){
        array[i] = gen_random();
    }
}

void fill_random_wide(wide_type* array, unsigned length, unsigned wide_length){
    for(unsigned i = 0; i < length; i++){
        for(unsigned j = 0; j < wide_length; j++){
            array[i][j] = gen_random();
        }
    }
}

void fill_random(float* array, unsigned length){
    for(unsigned i = 0; i < length; i++){
        array[i] = gen_random();
    }
}

void fill_with_value(float* array, float value, unsigned length){
    for(unsigned i = 0; i < length; i++){
        array[i] = value;
    }
}

void fill_with_zeros(HLSNN_DataType* array, unsigned length){
    for(unsigned i = 0; i < length; i++){
        array[i] = 0.0;
    }
}

void fill_with_zeros_wide(wide_type* array, unsigned length, unsigned wide_length){
    for(unsigned i = 0; i < length; i++){
        for(unsigned j = 0; j < wide_length; j++){
            array[i][j] = 0.0;
        }
    }
}

void fill_with_zeros_wide_vector(std::vector<wide_type> &array, unsigned length, unsigned wide_length){
    for(unsigned i = 0; i < length; i++){
        wide_type tmp;
        for(unsigned j = 0; j < wide_length; j++){
            tmp[j] = HLSNN_DataType(0.0);
        }
        array.push_back(tmp);
    }
}

void fill_with_values_wide_vector(std::vector<wide_type> &array, HLSNN_DataType value, unsigned length, unsigned wide_length){
    for(unsigned i = 0; i < length; i++){
        wide_type tmp;
        for(unsigned j = 0; j < wide_length; j++){
            tmp[j] = value;
        }
        array.push_back(tmp);
    }
}

void fill_with_values_wide(wide_type* array, HLSNN_DataType value, unsigned length, unsigned wide_length){
    for(unsigned i = 0; i < length; i++){
        for(unsigned j = 0; j < wide_length; j++){
            array[i][j] = value;      
        }
    }
}

template<unsigned F2>
void fill_FC_arrays_random(
    HLSNN_DataType **IFM,
    HLSNN_DataType **OFM,
    HLSNN_DataType **W,
    HLSNN_DataType **W_flip,
    HLSNN_DataType (&bias)[F2],
    unsigned b,
    unsigned cin,
    unsigned fout
){

    for(unsigned c = 0; c < cin; c++){
        for(unsigned f = 0; f < fout; f++){
            W[c][f] = gen_random();
            W_flip[f][c] = gen_random();
        }
    }

    for(unsigned m = 0; m < b; m++){
        for(unsigned c = 0; c < cin; c++){
            IFM[m][c] = gen_random();
        }
    }

    for(unsigned m = 0; m < b; m++){
        for(unsigned f = 0; f < fout; f++){
            OFM[m][f] = gen_random();
        }
    }
    
    for(unsigned f = 0; f < fout; f++){
        bias[f] = gen_random();
    }

}
