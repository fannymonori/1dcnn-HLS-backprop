#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <random>
#include <fstream>

#include "ap_fixed.h"
#include "hls_burst_maxi.h"
#include "hls_math.h"

#include "types.hpp"
#include "cnpy.h"

#define LEARNING_RATE 0.01

typedef float math_type;

//==============================================================================================================
// Functions for reading data from .npz files

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

//==============================================================================================================
// Functions for reading/writing/copying data from/to arrays

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

//==============================================================================================================
// Functions for printing

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

//==============================================================================================================
// Functions for neural network computations.

template<unsigned B_, unsigned F_>
static std::pair<float, float> compute_softmax(wide_type *inM, wide_type *outD, float *labels, unsigned B, unsigned F, unsigned wide_col, unsigned wide_length){

	math_type expsum_tmp[B_][F_];
    math_type epsilon = 0.00000001;
    math_type softmax_tmp[B_][F_];

	float loss;
	float expsum[B_];
    unsigned chunk_index = 0;
    for(int i = 0; i < B; i++){
    	math_type tmp = math_type(0.0);
        unsigned col_index = 0;
        for(int j = 0; j < wide_col; j++){
            for(int jj = 0; jj < wide_length; jj++){
                if(col_index < F){
        		    math_type v = (math_type) inM[chunk_index][col_index] + epsilon;
                    tmp += (std::exp(v));
        	    }
                col_index++;
            }
        }
        chunk_index++;
        expsum[i] = tmp;
    }

    //std::cout << "EXPSUM" << std::endl;
    //for(int i = 0; i < B; i++){
    //    std::cout << expsum[i] << std::endl;
    //}
    //std::cout << std::endl;

    float sum = 0.0;
    chunk_index = 0;
    for(int i = 0; i < B; i++){
        unsigned col_index = 0;
        for(int j = 0; j < wide_col; j++){
            for(int jj = 0; jj < wide_length; jj++){
                if(col_index < F){
                    float v = (float)inM[chunk_index][col_index];
                    softmax_tmp[i][col_index] = std::exp(v) / expsum[i];
                    sum += softmax_tmp[i][col_index];
                }
                else{
                    softmax_tmp[i][col_index] = float(0.0);
                }
                col_index++;
            }
        }
        chunk_index++;
    }

    // This is only for calculating accuracy and loss
    unsigned argmaxes[B_];
    for(int i = 0; i < B; i++){
        math_type max = 0.0;
        unsigned argmax = 0;
    	for(int j = 0; j < F; j++){
            if(i < B && j < F){
                if(max < math_type(softmax_tmp[i][j])){
                    max = math_type(softmax_tmp[i][j]);
                    argmax = j;
                }
            }
        }
        argmaxes[i] = argmax;        
    }    

    float acc = 0.0;
    math_type loss_sum = 0.0;
    unsigned vector_index = 0;
    for(int i = 0; i < B; i++){
    	for(int j = 0; j < F; j++){
            unsigned argmax = 0;
            math_type max_ = 0.0;
        	if(i < B && j < F){
                if(math_type(labels[vector_index]) > 0.5){
                    loss_sum += -1 * std::log(std::max(math_type(softmax_tmp[i][j]) * math_type(labels[vector_index]), epsilon));

                    if(argmaxes[i] == j){
                        acc += 1.0;
                    }
                }
                vector_index++;
        	}
        	else{
        	}
        }
    }

    loss_sum = loss_sum / float(B);
    std::cout << "LOSS: " << std::to_string(loss_sum) << std::endl;

    unsigned c = 0;
    chunk_index = 0;
    vector_index = 0;
    for(int i = 0; i < B; i++){
        unsigned col_index = 0;
    	for(int j = 0; j < wide_col; j++){
            for(int jj = 0; jj < wide_length; jj++){
                if(col_index < F){
                    outD[chunk_index][col_index] = HLSNN_DataType((softmax_tmp[i][col_index] - labels[vector_index]) / B);
                    vector_index++;
                }
                else{
                    outD[chunk_index][col_index] = HLSNN_DataType(0.0);
                }
                col_index++;
            }
        }
        chunk_index++;
    }

    return std::make_pair(loss_sum, acc);
}

void weight_sgd(std::vector<wide_type>& matrix, std::vector<wide_type>& grad, int M, int N, unsigned wide_length) {
    float lr = float(LEARNING_RATE);
    int total_elements = M * N;
    for (int i = 0; i < total_elements; ++i) {
        int tile_index = i / wide_length;
        int tile_element = i % wide_length;
        matrix[tile_index][tile_element] = HLSNN_DataType(float(matrix[tile_index][tile_element]) - float(lr) * float(grad[tile_index][tile_element]));
    }
}

void tanh_activation_bw(wide_type *tanh_dx_out, wide_type *tanh_in, wide_type *grad_in, int M, int N, unsigned wide_length) {
    int total_elements = M * N;
    for (int i = 0; i < total_elements; ++i) {
        int tile_index = i / wide_length;
        int tile_element = i % wide_length;
        float tanh_v = float(tanh_in[tile_index][tile_element]);
        float tanh_pow = 1.0 - (tanh_v * tanh_v);
        float grad = float(grad_in[tile_index][tile_element]);
        tanh_dx_out[tile_index][tile_element] = grad * tanh_pow;
    }
}

std::vector<HLSNN_DataType> sum_axis_0(std::vector<wide_type>& matrix, int M, int N, int wide_length) {
    std::vector<HLSNN_DataType> result(N, 0.0);
    for (int i = 0; i < M; ++i) {
        int row_offset = i * N;
        for (int j = 0; j < N; ++j) {
            int idx = row_offset + j;
            result[j] += matrix[idx / wide_length][idx % wide_length];
        }
    }
    return result;
}

void bias_sgd(std::vector<HLSNN_DataType>& vec, std::vector<HLSNN_DataType>& grad) {
    HLSNN_DataType lr_ = HLSNN_DataType(LEARNING_RATE);
    for (unsigned i = 0; i < grad.size(); i++) {
        vec[i] = vec[i] - lr_ * grad[i];
    }
}

//==============================================================================================================

extern "C"{
    void top_mm(
            hls::burst_maxi<wide_type> out_grad,
            hls::burst_maxi<wide_type> weight,
            hls::burst_maxi<wide_type> input_grad,
            HLSNN_DataType* bias,
            unsigned mode,
            unsigned int B, unsigned int C, unsigned int F, unsigned do_tanh
    );
}

#define B 32
#define M 128
#define N1 128
#define N2 6

#define W_SIZE 4*(M*M)
#define B_SIZE 2*(M)

#define OUT_SIZE 4*(B*M)

#define MAX_F N1
#define MAX_B B

#define MAX_OUT N2

int main(){

    std::cout << "Precision is: " << HLSNN_PRECISION << "," << HLSNN_INTPART << std::endl;

    char *cwd = get_current_dir_name();
    std::string working_dir(cwd);
    free(cwd);

    std::ofstream out_file;
    out_file.open (working_dir + "/output.txt");

	std::string weights_path = working_dir + "/gas_mlp_data_dX_full.npz";
	std::vector<std::string> layer_names = {"ograd", "w", "input", "bias", "w2", "bias2", "y"};
	std::map<std::string, std::vector<HLSNN_DataType>> dataMap;
	std::map<std::string, std::vector<double>> dataMap_float;

	std::cout << weights_path << std::endl;

    read_data_from_npz(weights_path, layer_names, dataMap, dataMap_float);

    std::cout << dataMap.size() << std::endl;

    std::vector<HLSNN_DataType> input_storage;
    input_storage.resize(B * 128, HLSNN_DataType(0.0));

    std::vector<wide_type> input_storage_wide;

    std::vector<HLSNN_DataType> bias_storage;
    bias_storage.resize(128, HLSNN_DataType(0.0));

    std::vector<HLSNN_DataType> bias2_storage;
    bias2_storage.resize(128, HLSNN_DataType(0.0));

    //std::vector<HLSNN_DataType> weightStorage_1;
    //weightStorage_1.resize(128*128, HLSNN_DataType(0.0));

    std::vector<wide_type> weightStorage_wide;

    std::vector<wide_type> weightStorage2_wide;

    std::vector<wide_type> weightStorage_wide_dW;
    weightStorage_wide_dW.resize(128*128);

    std::vector<wide_type> weightStorage_wide_tr;
    weightStorage_wide_tr.resize(128*128);

    HLSNN_DataType* w_ptr = dataMap["w"].data();
    HLSNN_DataType* bias_ptr = dataMap["bias"].data();
    HLSNN_DataType* w2_ptr = dataMap["w2"].data();
    HLSNN_DataType* bias2_ptr = dataMap["bias2"].data();

    unsigned c = 0;
    unsigned c2 = 0;

    // READ B1
    unsigned s = 0;
    for (int j = 0; j < N1; j++) {
    	bias_storage[j] = bias_ptr[j];
    }

	//for(int j = 0; j < M * N1; j++){
	//	weightStorage_1[j] = w_ptr[j];
	//}

    // READ W1
	c = 0;
	for(int i = 0; i < M; i++){
		for(int j = 0; j < N1/WIDE_LEN; j++){

			wide_type tmp;
			for(int k = 0; k < WIDE_LEN; k++){
				HLSNN_DataType v = w_ptr[c];
				tmp[k] = v;
				c++;
			}

			weightStorage_wide.push_back(tmp);
		}
	}

    // READ b2
	for (int j = 0; j < N2; j++) {
		bias2_storage[j] = bias2_ptr[j];
	}

	// READ W2
    unsigned loop_bound = N2 > WIDE_LEN? N2 / WIDE_LEN : 1;
	c = 0;
	for(int i = 0; i < N1; i++){
		for(int j = 0; j < loop_bound; j++){

			wide_type tmp;
			for(int k = 0; k < WIDE_LEN; k++){
				if(k < N2){
					HLSNN_DataType v = w2_ptr[c];
					tmp[k] = v;
					c++;
				}
				else{
					tmp[k] = 0.0;
				}
			}

			weightStorage2_wide.push_back(tmp);
		}
	}


    out_file << "Weight 1:" << std::endl;
    print_output_wide(weightStorage_wide.data(), M, N1, WIDE_LEN, out_file);
    out_file << "========================================================" << std::endl;
    out_file << "Weight 2:" << std::endl;
    print_output_wide(weightStorage2_wide.data(), N1, N2, WIDE_LEN, out_file);
    out_file << "========================================================" << std::endl;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////

    unsigned int b = unsigned(B);
    unsigned int m = unsigned(M);
    unsigned int n1 = unsigned(N1);
    unsigned int n2 = unsigned(N2);

	std::vector<HLSNN_DataType> x_grad_storage;
	x_grad_storage.resize(MAX_B * 128, HLSNN_DataType(0.0));

    std::vector<HLSNN_DataType> h1_storage;
    h1_storage.resize(MAX_B * 128, HLSNN_DataType(0.0));

    std::vector<wide_type> h1_storage_wide;
    h1_storage_wide.resize(MAX_B * 128, HLSNN_DataType(0.0));

    std::vector<HLSNN_DataType> h1_grad_storage;
    h1_grad_storage.resize(MAX_B * 128, HLSNN_DataType(0.0));

    std::vector<wide_type> h1_grad_storage_wide;
    h1_grad_storage_wide.resize(MAX_B * 128, HLSNN_DataType(0.0));

    std::vector<wide_type> r1_grad_storage_wide;
    r1_grad_storage_wide.resize(MAX_B * 128, HLSNN_DataType(0.0));

    std::vector<HLSNN_DataType> h2_storage;
    h2_storage.resize(MAX_B * 128, HLSNN_DataType(0.0));

    std::vector<wide_type> h2_storage_wide;
    h2_storage_wide.resize(MAX_B * 128, HLSNN_DataType(0.0));

    std::vector<wide_type> weightStorage_wide2_dW;
    weightStorage_wide2_dW.resize(128*128);

    std::vector<wide_type> weightStorage_wide2_tr;
    weightStorage_wide2_tr.resize(128*128);

    std::vector<HLSNN_DataType> out_grad_;
    out_grad_.resize(MAX_B * 128, HLSNN_DataType(0.0));

    std::vector<wide_type> outgrad_wide;

    std::vector<wide_type> outgrad_wide_tmp;

    unsigned o_i = 0;
    for(int j = 0; j < MAX_B * MAX_F; j++){
    	x_grad_storage[j] = 0.0;
    	h1_storage[j] = 0.0;
    	h2_storage[j] = 0.0;
    }

    std::cout << weightStorage_wide.size() << std::endl;
    std::cout << input_storage_wide.size() << std::endl;

    //unsigned num_of_batches = 14;
    unsigned num_of_batches = 1;
    //unsigned number_of_epochs = 25;
    unsigned number_of_epochs = 1;
    //unsigned number_of_epochs = 1;

    std::vector<float> training_accuracies;
    std::vector<float> training_losses;

    for(int ee = 0; ee < number_of_epochs; ee++){

        std::vector<std::vector<wide_type>> inputs_vector;
        std::vector<std::vector<float>> labels_vector;
        for(int bb = 0; bb < num_of_batches; bb++){


            std::string input_path = working_dir + "/Data/v2/epoch_" + std::to_string(ee) + "/" + std::to_string(bb) + ".npz";
            std::cout << input_path << std::endl;
            std::vector<std::string> layer_names_batch = {"input", "y"};
	        std::map<std::string, std::vector<double>> dataMap_float_batch;
            std::map<std::string, std::vector<HLSNN_DataType>> dataMap_batch;
            read_data_from_npz(input_path, layer_names_batch, dataMap_batch, dataMap_float_batch);
            HLSNN_DataType* x_ptr = dataMap_batch["input"].data();

            std::vector<wide_type> input_storage_wide;
            place_in_wide(x_ptr, input_storage_wide, B, N1, WIDE_LEN);

            HLSNN_DataType* y = dataMap_batch["y"].data();
            std::vector<float> labels;
            unsigned l = 0;
            for(int i = 0; i < B; i++){
                for(int j = 0; j < N2; j++){              
                    labels.push_back(float(y[l]));
                    l++;
                }
            }

            inputs_vector.push_back(input_storage_wide);
            labels_vector.push_back(labels);
        }

        float running_acc = 0.0;
        float running_acc2 = 0.0;
        float running_loss = 0.0;
        for(int bb = 0; bb < num_of_batches; bb++){

            for(int j = 0; j < MAX_B * MAX_F; j++){
    	        x_grad_storage[j] = 0.0;
    	        h1_storage[j] = 0.0;
    	        h2_storage[j] = 0.0;
            }

            outgrad_wide.clear();
            out_grad_.resize(MAX_B * 128, HLSNN_DataType(0.0));

            outgrad_wide.resize(MAX_B * 8);

            weightStorage_wide2_dW.clear();
            weightStorage_wide2_dW.resize(128*128);

            weightStorage_wide_dW.clear();
            weightStorage_wide_dW.resize(128*128);

            // FW layers fully on FPGA
            //std::cout << std::endl << "################################## FW layer 1 ##################################" << std::endl;

            //FW
            top_mm(
                            h1_storage_wide.data(),
                            weightStorage_wide.data(),
                            inputs_vector[bb].data(),
                            bias_storage.data(),
                            0,
                            b, m, n1, 1
            );


            if(ee == 0 && bb == 0){
                out_file << "========================================================" << std::endl;
                out_file << "H1:" << std::endl;
                print_output_wide(h1_storage_wide.data(), B, N1, WIDE_LEN, out_file);
                out_file << "========================================================" << std::endl;
            }

            //std::cout << std::endl << "################################## FW layer 2 ##################################" << std::endl;
            top_mm(
                            h2_storage_wide.data(),
                            weightStorage2_wide.data(),
                            h1_storage_wide.data(),
                            bias2_storage.data(),
                            0,
                            b, n1, n2, 0
            );

            //std::cout << std::endl << "##################################  SOFTMAX  ##################################" << std::endl;

            if(ee == 0 && bb == 0){
                out_file << "H2:" << std::endl;
                print_output_wide(h2_storage_wide.data(), B, N2, WIDE_LEN, out_file);
                out_file << "========================================================" << std::endl;
            }

            float loss, acc;
            std::tie(loss, acc) = compute_softmax<B, N2>(
                    h2_storage_wide.data(),
                    outgrad_wide.data(),
                    labels_vector[bb].data(),
                    b, n2, 1, WIDE_LEN);

            acc = acc / 32.0;

            running_acc += acc;
            running_loss += loss;
            std::cout << loss << std::endl;

            //std::cout << std::endl << "##################################  dX layer 2  ##################################" << std::endl;
            //////////////// dX

            top_mm(
                            outgrad_wide.data(),
                            weightStorage2_wide.data(),
                            r1_grad_storage_wide.data(),
                            bias2_storage.data(),
                            1,
                            b, n1, n2, 0
            );

            tanh_activation_bw(h1_grad_storage_wide.data(), h1_storage_wide.data(), r1_grad_storage_wide.data(), B, N1, WIDE_LEN);

            ///////////////
            //std::cout << std::endl << "##################################  dW layer 2  ##################################" << std::endl;

            top_mm(
                        outgrad_wide.data(),
                        weightStorage_wide2_dW.data(),
                        h1_storage_wide.data(),
                        bias2_storage.data(),
                        2,
                        b, n1, n2, 0
            );


            //std::cout << std::endl << "##################################  dW layer 1  ##################################" << std::endl;
            ///##############

            top_mm(
                        h1_grad_storage_wide.data(),
                        weightStorage_wide_dW.data(),
                        inputs_vector[bb].data(),
                        bias_storage.data(),
                        2,
                        b, m, n1, 0
            );

            std::vector<HLSNN_DataType> db2 = sum_axis_0(outgrad_wide, B, WIDE_LEN, WIDE_LEN);
            bias_sgd(bias2_storage, db2);
            std::vector<HLSNN_DataType> db1 = sum_axis_0(h1_grad_storage_wide, B, N1, WIDE_LEN);
            bias_sgd(bias_storage, db1);

            weight_sgd(weightStorage_wide, weightStorage_wide_dW, M, N1, WIDE_LEN);
            weight_sgd(weightStorage2_wide, weightStorage_wide2_dW, N1, WIDE_LEN, WIDE_LEN);

            if(ee == 0 && bb == 0){
                out_file << "========================================================" << std::endl;
                out_file << "OGRAD:" << std::endl;
                print_output_wide(outgrad_wide.data(), B, WIDE_LEN, WIDE_LEN, out_file);
                out_file << "========================================================" << std::endl;
                out_file << "B2 UPDATED:" << std::endl;
                print_output(bias2_storage.data(), 1, WIDE_LEN, out_file);
                out_file << "========================================================" << std::endl;
                out_file << "DR1:" << std::endl;
                print_output_wide(r1_grad_storage_wide.data(), B, N1, WIDE_LEN, out_file);
                out_file << "========================================================" << std::endl;
                out_file << "DH1:" << std::endl;
                print_output_wide(h1_grad_storage_wide.data(), B, N1, WIDE_LEN, out_file);
                out_file << "========================================================" << std::endl;
                out_file << "B1 UPDATED:" << std::endl;
                print_output(bias_storage.data(), 1, N1, out_file);
                out_file << "========================================================" << std::endl;
                out_file << "dW1:" << std::endl;
                print_output_wide(weightStorage_wide_dW.data(), M, N1, WIDE_LEN, out_file);
                out_file << "========================================================" << std::endl;
                out_file << "dW2:" << std::endl;
                print_output_wide(weightStorage_wide2_dW.data(), N1, WIDE_LEN, WIDE_LEN, out_file);
                out_file << "========================================================" << std::endl;
                out_file << "W1 updated:" << std::endl;
                print_output_wide(weightStorage_wide.data(), M, N1, WIDE_LEN, out_file);
                out_file << "========================================================" << std::endl;
                out_file << "W2 updated:" << std::endl;
                print_output_wide(weightStorage2_wide.data(), N1, WIDE_LEN, WIDE_LEN, out_file);
                out_file << "========================================================" << std::endl;
            }
        }

        //running_acc = (running_acc / float(num_of_batches * 32)) * 100.0;
        running_acc = (running_acc / float(num_of_batches)) * 100.0;
        std::cout << "Accuracy is: " << running_acc << std::endl;
        training_accuracies.push_back(running_acc);

        running_loss = running_loss / float(num_of_batches);
        std::cout << "Loss is: " << running_loss << std::endl;
        training_losses.push_back(running_loss);
    }


    std::cout << "\n TRAINING ACCURACIES \n" << std::endl;
    for(const float& i : training_accuracies){
        std::cout << std::to_string(i) << ",";
    }
    std::cout << std::endl;

    std::cout << "\n TRAINING LOSSES \n" << std::endl;
    for(const float& i : training_losses){
        std::cout << std::to_string(i) << ",";
    }
    std::cout << std::endl;

    return 0;

    std::vector<float> testing_accuracies;
    unsigned num_of_batches_array_test[9] = {39, 50, 6, 7, 72, 113, 10, 15, 113};
    for(int ee = 1; ee < 10; ee++){

        std::vector<std::vector<wide_type>> inputs_vector;
        std::vector<std::vector<float>> labels_vector;
        for(int bb = 0; bb < num_of_batches_array_test[ee-1]; bb++){


            std::string input_path = working_dir + "/Data/v2/test_" + std::to_string(ee) + "/" + std::to_string(bb) + ".npz";
            std::cout << input_path << std::endl;
            std::vector<std::string> layer_names_batch = {"input", "y"};
	        std::map<std::string, std::vector<double>> dataMap_float_batch;
            std::map<std::string, std::vector<HLSNN_DataType>> dataMap_batch;
            read_data_from_npz(input_path, layer_names_batch, dataMap_batch, dataMap_float_batch);
            HLSNN_DataType* x_ptr = dataMap_batch["input"].data();

            std::vector<wide_type> input_storage_wide;
            place_in_wide(x_ptr, input_storage_wide, B, N1, WIDE_LEN);

            HLSNN_DataType* y = dataMap_batch["y"].data();
            std::vector<float> labels;
            unsigned l = 0;
            for(int i = 0; i < B; i++){
                for(int j = 0; j < N2; j++){              
                    labels.push_back(float(y[l]));
                    l++;
                }
            }

            inputs_vector.push_back(input_storage_wide);
            labels_vector.push_back(labels);
        }


        float running_acc = 0.0;
        for(int bb = 0; bb < num_of_batches_array_test[ee-1]; bb++){

            top_mm(
                            h1_storage_wide.data(),
                            weightStorage_wide.data(),
                            inputs_vector[bb].data(),
                            bias_storage.data(),
                            0,
                            b, m, n1, 0
            );

            //std::cout << std::endl << "################################## FW layer 2 ##################################" << std::endl;
            top_mm(
                            h2_storage_wide.data(),
                            weightStorage2_wide.data(),
                            h1_storage_wide.data(),
                            bias2_storage.data(),
                            0,
                            b, n1, n2, 0
            );

            //std::cout << std::endl << "##################################  SOFTMAX  ##################################" << std::endl;

            float loss, acc;
            std::tie(loss, acc) = compute_softmax<B, N2>(
                    h2_storage_wide.data(),
                    outgrad_wide.data(),
                    labels_vector[bb].data(),
                    b, n2, 1, WIDE_LEN);

            running_acc += acc;
        }

        running_acc = (running_acc / float(num_of_batches_array_test[ee-1] * 32)) * 100.0;
        std::cout << "Accuracy is: " << running_acc << std::endl;

        testing_accuracies.push_back(running_acc);
    }
    

    ///////////////

    ///////////////

    std::cout << " ==OUTPUTS======================================================================== " << std::endl;
    std::cout << "\n RESULTS \n" << std::endl;


    std::cout << "\n TRAINING ACCURACIES \n" << std::endl;
    for(const float& i : training_accuracies){
        std::cout << std::to_string(i) << ",";
    }
    std::cout << std::endl;

    std::cout << "\n TESTING ACCURACIES \n" << std::endl;
    for(const float& i : testing_accuracies){
        std::cout << std::to_string(i) << ",";
    }
    std::cout << std::endl;


    return 0;
}