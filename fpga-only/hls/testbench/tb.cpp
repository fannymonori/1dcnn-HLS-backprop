#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <inttypes.h>
#include <map>
#include <unistd.h>
#include <chrono>
#include <cmath>

#include <ap_int.h>
#include "ap_fixed.h"

#include "types.hpp"
#include "hlsnn_matrix.hpp"
#include "hls_math.h"

template <typename T, typename T2, unsigned int N>
T CrossEntropyLoss(hlsnn::matrix<T, 1, N>& pred, hlsnn::matrix<T2, 1, N>& gt) {
    T sum = 0;
    for (int i = 0; i < N; i++) {
        sum += gt(0, i) * ap_fixed<64, 32>(hls::log((float)pred(0, i)));
    }

    return -1 * sum;
}

#define K 2
#define M 100
#define N 6

#define conv_cout 32

#define conv_2_cin 32
#define conv_2_cout 16

#define conv_3_cin 16
#define conv_3_cout 16

#define conv_4_cin 16
#define conv_4_cout 16

#define conv_5_cin 16
#define conv_5_cout 16

#define MP_1_M 50
#define MP_2_M 25

#define D_in 400
#define D_out 3

#define pad (K - 1)
#define M_padded (M + pad)

#define MAX_EPOCH 100

#define Cin N
#define Cout 8

#define Cin2 Cout
#define Cout2 10

#define MP_M 50

#define D_out 3

#define pad (K - 1)
#define M_padded (M + pad)

extern "C"{
void dcnn1d_top_orig(
        HLSNN_DataType input_sample[M * N],
		HLSNN_DataType result[D_out],
		int label_data[D_out],
		bool training_enabled,
        bool finished_training[1],
        bool startup
		);
}

void run_train(std::string &data_file){


    std::vector<std::vector<int>> labels;
    std::vector<std::vector<HLSNN_DataType>> data;

    std::string line, s;

    std::ifstream sample_file(data_file);
    if (sample_file.is_open()){
    	int count = 0;

    	while (std::getline(sample_file, line))
    	{
    		std::istringstream ss(line);

    		if(count % 2 == 0){
    			// process label
    			std::vector<int> label;
    			while (getline(ss, s, ' ')) {
    				label.push_back(std::stoi(s));
    			}
    				labels.push_back(label);
            }
    		else{
    			// process data
    			std::vector<HLSNN_DataType> sample;
    			while (getline(ss, s, ' ')) {
    				sample.push_back(std::stof(s));
    			}
    			data.push_back(sample);
    		}


    		count++;
    	}
    }
	else{
		std::cout << "Could not open file! " << data_file << std::endl;
	}

	int data_size = data.size();

    std::cout << "\n\n\n" << std::endl;

    HLSNN_DataType result[D_out];
    hlsnn::matrix<HLSNN_DataType, 1, D_out> out_softmax_pl;

    bool initialize = 1;
    //for(int d = 0; d < MAX_EPOCH; d++){
    int setup = false;
    for(int d = 0; d < 3; d++){

        if(d == 0){
            setup = true;
        }
        else{
            setup = false;
        }

    	hlsnn::matrix<int, 1, D_out> label(labels[d].data());

        bool finish_flag[1];
        finish_flag[0] = false;

        for(int i = 0; i < D_out; i++){
            result[i] = 0;
        }

    	dcnn1d_top_orig(data[d].data(),
				result,
				labels[d].data(),
                true,
                finish_flag,
                setup);

    	out_softmax_pl.loadData(result);
        for(int i = 0; i < D_out; i++){
            std::cout << result[i] << " ";
        }
        std::cout << std::endl;

        double loss = CrossEntropyLoss(out_softmax_pl, label);
    	std::cout << "Loss from PL inference: " << loss << " Training finished: " << finish_flag[0] << std::endl;

    	initialize = false;
    }
}

int run(){

    char *cwd = get_current_dir_name();
    std::string working_dir(cwd);
    free(cwd);

	std::string data_file = "/train_samples_wine.txt";
    std::string path_to_file = working_dir + data_file;

	run_train(path_to_file);

	return 0;
}

int main(){
	run();

	return 0;
}
