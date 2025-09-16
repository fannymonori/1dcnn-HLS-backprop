#include "types.hpp"
#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }


#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <CL/cl2.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <chrono>
#include <pthread.h>
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>
#include "cnpy.h"
#include <fstream>


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

#define L_RATE 0.001

int run_on_CPU_FPGA(std::string xclbinFilename_, std::string pathToData_, cl::Context &context, cl::CommandQueue &q, cl::Program &program) {

    std::string xclbinFilename = xclbinFilename_;
    std::string pathToData = pathToData_;

    std::ofstream out_file;
    out_file.open ("./output.txt");

    cl_int err;

    //////////////////////////

	std::string weights_path = pathToData + "/gas_mlp_data_dX_full.npz";
	std::vector<std::string> layer_names = {"ograd", "w", "input", "bias", "w2", "bias2", "y"};
	std::map<std::string, std::vector<HLSNN_DataType>> dataMap;
	std::map<std::string, std::vector<double>> dataMap_float;

	std::cout << weights_path << std::endl;

    read_data_from_npz(weights_path, layer_names, dataMap, dataMap_float);

    std::cout << dataMap.size() << std::endl;

    std::vector<HLSNN_DataType> input_storage;
    std::vector<wide_type> input_storage_wide;
    std::vector<HLSNN_DataType> bias_storage;
    std::vector<HLSNN_DataType> bias2_storage;
    std::vector<HLSNN_DataType> weightStorage_1;
    std::vector<wide_type> weightStorage_wide;
    std::vector<wide_type> weightStorage2_wide;
    std::vector<wide_type> weightStorage_wide_dW;
    std::vector<wide_type> weightStorage_wide_tr;

    input_storage.resize(B * 128, HLSNN_DataType(0.0));
    bias_storage.resize(128, HLSNN_DataType(0.0));
    bias2_storage.resize(128, HLSNN_DataType(0.0));
    weightStorage_1.resize(128*128, HLSNN_DataType(0.0));
    weightStorage_wide_dW.resize(128*128);
    weightStorage_wide_tr.resize(128*128);

    HLSNN_DataType* w_ptr = dataMap["w"].data();
    HLSNN_DataType* bias_ptr = dataMap["bias"].data();
    HLSNN_DataType* w2_ptr = dataMap["w2"].data();
    HLSNN_DataType* bias2_ptr = dataMap["bias2"].data();

    unsigned s = 0;
    for (int j = 0; j < N1; j++) {
    	bias_storage[j] = bias_ptr[j];
    }

    for(int j = 0; j < M * N1; j++){
		weightStorage_1[j] = w_ptr[j];
	}

	unsigned c = 0;
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

	for (int j = 0; j < N2; j++) {
		bias2_storage[j] = bias2_ptr[j];
	}

    out_file << "Weight 1:" << std::endl;
    print_output_wide(weightStorage_wide.data(), M, N1, WIDE_LEN, out_file);
    out_file << "========================================================" << std::endl;
    out_file << "Weight 2:" << std::endl;
    print_output_wide(weightStorage2_wide.data(), N1, N2, WIDE_LEN, out_file);
    out_file << "========================================================" << std::endl;


    ///////////////////////////////////////////////////////////// SET UP DEVICE

    cl::Kernel krnl_fw1, krnl_fw2, krnl_tanh, krnl_dx2, krnl_dw2, krnl_dw1, krnl_wu, krnl_wu2, krnl_tanh_bw;
    OCL_CHECK(err, krnl_fw1 = cl::Kernel(program, "top_mm", &err));
    OCL_CHECK(err, krnl_fw2 = cl::Kernel(program, "top_mm", &err));
    OCL_CHECK(err, krnl_dx2 = cl::Kernel(program, "top_mm", &err));
    OCL_CHECK(err, krnl_dw2 = cl::Kernel(program, "top_mm", &err));
    OCL_CHECK(err, krnl_dw1 = cl::Kernel(program, "top_mm", &err));
    OCL_CHECK(err, krnl_wu = cl::Kernel(program, "top_wu", &err));
    OCL_CHECK(err, krnl_wu2 = cl::Kernel(program, "top_wu", &err));
    OCL_CHECK(err, krnl_tanh = cl::Kernel(program, "tanh_top", &err));
    OCL_CHECK(err, krnl_tanh_bw = cl::Kernel(program, "tanh_top", &err));

    unsigned int b = unsigned(B);
    unsigned int m = unsigned(M);
    unsigned int n1 = unsigned(N1);
    unsigned int n2 = unsigned(N2);

    unsigned ROW_SIZE = 64;
    unsigned DATA_SIZE = ROW_SIZE*ROW_SIZE;
    unsigned WIDE_SIZE = DATA_SIZE/WIDE_LEN;

    OCL_CHECK(err, cl::Buffer buffer_x(context, CL_MEM_READ_WRITE, (MAX_B * MAX_F) * sizeof(wide_type), NULL, &err));
    
    OCL_CHECK(err, cl::Buffer buffer_w1(context, CL_MEM_READ_WRITE, (MAX_F * MAX_F) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_w2(context, CL_MEM_READ_WRITE, (MAX_F * MAX_F) * sizeof(wide_type), NULL, &err));

    OCL_CHECK(err, cl::Buffer buffer_b1(context, CL_MEM_READ_WRITE, (MAX_F) * sizeof(HLSNN_DataType), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_b2(context, CL_MEM_READ_WRITE, (MAX_F) * sizeof(HLSNN_DataType), NULL, &err));
    
    OCL_CHECK(err, cl::Buffer buffer_h1(context, CL_MEM_READ_WRITE, (MAX_B * MAX_F) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_a1(context, CL_MEM_READ_WRITE, (MAX_B * MAX_F) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_h2(context, CL_MEM_READ_WRITE, (MAX_B * MAX_F) * sizeof(wide_type), NULL, &err));

    OCL_CHECK(err, cl::Buffer buffer_out_grad(context, CL_MEM_READ_WRITE, (MAX_B * MAX_F) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dx_l2(context, CL_MEM_READ_WRITE, (MAX_B * MAX_F) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dx_al2(context, CL_MEM_READ_WRITE, (MAX_B * MAX_F) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dw_l2(context, CL_MEM_READ_WRITE, (MAX_F * MAX_F) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dw_l1(context, CL_MEM_READ_WRITE, (MAX_F * MAX_F) * sizeof(wide_type), NULL, &err));

    OCL_CHECK(err, cl::Buffer buffer_tmp1(context, CL_MEM_READ_WRITE, (B*(N1/WIDE_LEN)) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_tmp2(context, CL_MEM_READ_WRITE, (B*(N1/WIDE_LEN)) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_tmp3(context, CL_MEM_READ_WRITE, (B*(N1/WIDE_LEN)) * sizeof(wide_type), NULL, &err));

    unsigned mode = 0;

    // set the kernel Arguments
    int narg = 0;
    OCL_CHECK(err, err = krnl_fw1.setArg(narg++, buffer_h1));
    OCL_CHECK(err, err = krnl_fw1.setArg(narg++, buffer_w1));
    OCL_CHECK(err, err = krnl_fw1.setArg(narg++, buffer_x));
    OCL_CHECK(err, err = krnl_fw1.setArg(narg++, buffer_b1));
    OCL_CHECK(err, err = krnl_fw1.setArg(narg++, 0));
    OCL_CHECK(err, err = krnl_fw1.setArg(narg++, b));
    OCL_CHECK(err, err = krnl_fw1.setArg(narg++, m));
    OCL_CHECK(err, err = krnl_fw1.setArg(narg++, n1));
    OCL_CHECK(err, err = krnl_fw1.setArg(narg++, 0));

    narg = 0;
    OCL_CHECK(err, err = krnl_tanh.setArg(narg++, buffer_h1));
    OCL_CHECK(err, err = krnl_tanh.setArg(narg++, buffer_a1));
    OCL_CHECK(err, err = krnl_tanh.setArg(narg++, buffer_dx_l2));
    OCL_CHECK(err, err = krnl_tanh.setArg(narg++, unsigned(B*N1)));
    OCL_CHECK(err, err = krnl_tanh.setArg(narg++, true));

    narg = 0;
    OCL_CHECK(err, err = krnl_fw2.setArg(narg++, buffer_h2));
    OCL_CHECK(err, err = krnl_fw2.setArg(narg++, buffer_w2));
    OCL_CHECK(err, err = krnl_fw2.setArg(narg++, buffer_a1));
    OCL_CHECK(err, err = krnl_fw2.setArg(narg++, buffer_b2));
    OCL_CHECK(err, err = krnl_fw2.setArg(narg++, 0));
    OCL_CHECK(err, err = krnl_fw2.setArg(narg++, b));
    OCL_CHECK(err, err = krnl_fw2.setArg(narg++, n1));
    OCL_CHECK(err, err = krnl_fw2.setArg(narg++, n2));
    OCL_CHECK(err, err = krnl_fw2.setArg(narg++, 0));

    narg = 0;
    OCL_CHECK(err, err = krnl_dx2.setArg(narg++, buffer_out_grad));
    OCL_CHECK(err, err = krnl_dx2.setArg(narg++, buffer_w2));
    OCL_CHECK(err, err = krnl_dx2.setArg(narg++, buffer_dx_l2));
    OCL_CHECK(err, err = krnl_dx2.setArg(narg++, buffer_b2));
    OCL_CHECK(err, err = krnl_dx2.setArg(narg++, 1));
    OCL_CHECK(err, err = krnl_dx2.setArg(narg++, b));
    OCL_CHECK(err, err = krnl_dx2.setArg(narg++, n1));
    OCL_CHECK(err, err = krnl_dx2.setArg(narg++, n2));
    OCL_CHECK(err, err = krnl_dx2.setArg(narg++, 0));

    narg = 0;
    OCL_CHECK(err, err = krnl_dw2.setArg(narg++, buffer_out_grad));
    OCL_CHECK(err, err = krnl_dw2.setArg(narg++, buffer_dw_l2));
    OCL_CHECK(err, err = krnl_dw2.setArg(narg++, buffer_a1));
    OCL_CHECK(err, err = krnl_dw2.setArg(narg++, buffer_b2));
    OCL_CHECK(err, err = krnl_dw2.setArg(narg++, 2));
    OCL_CHECK(err, err = krnl_dw2.setArg(narg++, b));
    OCL_CHECK(err, err = krnl_dw2.setArg(narg++, n1));
    OCL_CHECK(err, err = krnl_dw2.setArg(narg++, n2));
    OCL_CHECK(err, err = krnl_dw2.setArg(narg++, 0));

    narg = 0;
    OCL_CHECK(err, err = krnl_dw1.setArg(narg++, buffer_dx_al2));
    OCL_CHECK(err, err = krnl_dw1.setArg(narg++, buffer_dw_l1));
    OCL_CHECK(err, err = krnl_dw1.setArg(narg++, buffer_x));
    OCL_CHECK(err, err = krnl_dw1.setArg(narg++, buffer_b2));
    OCL_CHECK(err, err = krnl_dw1.setArg(narg++, 2));
    OCL_CHECK(err, err = krnl_dw1.setArg(narg++, b));
    OCL_CHECK(err, err = krnl_dw1.setArg(narg++, m));
    OCL_CHECK(err, err = krnl_dw1.setArg(narg++, n1));
    OCL_CHECK(err, err = krnl_dw1.setArg(narg++, 0));

    narg = 0;

    narg = 0;
    OCL_CHECK(err, err = krnl_wu.setArg(narg++, buffer_w1));
    OCL_CHECK(err, err = krnl_wu.setArg(narg++, buffer_dw_l1));
    OCL_CHECK(err, err = krnl_wu.setArg(narg++, unsigned(M*N1)));

    narg = 0;
    OCL_CHECK(err, err = krnl_wu2.setArg(narg++, buffer_w2));
    OCL_CHECK(err, err = krnl_wu2.setArg(narg++, buffer_dw_l2));
    OCL_CHECK(err, err = krnl_wu2.setArg(narg++, unsigned(N1*WIDE_LEN)));

    wide_type* ptr_x, *ptr_w1, *ptr_w2, *ptr_h1, *ptr_a1, *ptr_h2, *ptr_outgrad, *ptr_dx_l2, *ptr_dw_l2, *ptr_dw_l1, *ptr_dx_al2;
    HLSNN_DataType *ptr_b1, *ptr_b2;
    OCL_CHECK(err, ptr_x = (wide_type*)q.enqueueMapBuffer(buffer_x, CL_TRUE, CL_MAP_WRITE, 0, (MAX_B * MAX_F) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, ptr_w1 = (wide_type*)q.enqueueMapBuffer(buffer_w1, CL_TRUE, CL_MAP_WRITE, 0, (MAX_F * MAX_F) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, ptr_w2 = (wide_type*)q.enqueueMapBuffer(buffer_w2, CL_TRUE, CL_MAP_WRITE, 0, (MAX_F * MAX_F) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, ptr_b1 = (HLSNN_DataType*)q.enqueueMapBuffer(buffer_b1, CL_TRUE, CL_MAP_WRITE, 0, (MAX_F) * sizeof(HLSNN_DataType), NULL, NULL, &err));
    OCL_CHECK(err, ptr_b2 = (HLSNN_DataType*)q.enqueueMapBuffer(buffer_b2, CL_TRUE, CL_MAP_WRITE, 0, (MAX_F) * sizeof(HLSNN_DataType), NULL, NULL, &err));
    
    OCL_CHECK(err, ptr_h1 = (wide_type*)q.enqueueMapBuffer(buffer_h1, CL_TRUE, CL_MAP_READ, 0, (MAX_B * MAX_F) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, ptr_a1 = (wide_type*)q.enqueueMapBuffer(buffer_a1, CL_TRUE, CL_MAP_READ, 0, (MAX_B * MAX_F) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, ptr_h2 = (wide_type*)q.enqueueMapBuffer(buffer_h2, CL_TRUE, CL_MAP_READ, 0, (MAX_B * MAX_F) * sizeof(wide_type), NULL, NULL, &err));
    
    OCL_CHECK(err, ptr_outgrad = (wide_type*)q.enqueueMapBuffer(buffer_out_grad, CL_TRUE, CL_MAP_READ, 0, (MAX_B * MAX_F) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, ptr_dx_l2 = (wide_type*)q.enqueueMapBuffer(buffer_dx_l2, CL_TRUE, CL_MAP_READ, 0, (MAX_B * MAX_F) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, ptr_dx_al2 = (wide_type*)q.enqueueMapBuffer(buffer_dx_al2, CL_TRUE, CL_MAP_READ, 0, (MAX_B * MAX_F) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, ptr_dw_l2 = (wide_type*)q.enqueueMapBuffer(buffer_dw_l2, CL_TRUE, CL_MAP_READ, 0, (MAX_F * MAX_F) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, ptr_dw_l1 = (wide_type*)q.enqueueMapBuffer(buffer_dw_l1, CL_TRUE, CL_MAP_READ, 0, (MAX_F * MAX_F) * sizeof(wide_type), NULL, NULL, &err));

    for(int j = 0; j < M * (N1 / WIDE_LEN); j++){
		ptr_w1[j] = weightStorage_wide[j];
	}

    for(int j = 0; j < N1; j++){
		ptr_w2[j] = weightStorage2_wide[j];
	}

    for(int j = 0; j < N1; j++){
		ptr_b1[j] = bias_storage[j];
	}

    for(int j = 0; j < N2; j++){
		ptr_b2[j] = bias2_storage[j];
	}

    fill_with_zeros_wide(ptr_outgrad, B, WIDE_LEN);
    fill_with_zeros_wide(ptr_dw_l1, M * (N1/WIDE_LEN), WIDE_LEN);
    fill_with_zeros_wide(ptr_dw_l2, N1, WIDE_LEN);

    std::vector<pthread_t> sgd_threads;
    std::vector<SGD_thread_data_wide*> sgd_data_vector;
    unsigned threading_factor = 4;
    unsigned lr_ = 0.001;
    unsigned w_wide_len = M * (N1 / WIDE_LEN);
    unsigned array_in_4 = int(w_wide_len / threading_factor);
    for(unsigned i = 0; i < threading_factor; ++i){
        unsigned start = i * array_in_4;
        unsigned end = start + array_in_4;
        struct SGD_thread_data_wide *sgd_data = (struct SGD_thread_data_wide *)malloc(sizeof(struct SGD_thread_data_wide));
        sgd_data->w_orig = ptr_w1;
        sgd_data->w = ptr_dw_l1;
        sgd_data->LR = lr_;
        sgd_data->length = w_wide_len;
        sgd_data->start = start;
        sgd_data->end = end;
        sgd_data->wide_length = WIDE_LEN;
        pthread_t sgd_thrd;
        sgd_data_vector.push_back(sgd_data);
        sgd_threads.push_back(sgd_thrd);
    }

    std::vector<pthread_t> sgd_threads_l2;
    std::vector<SGD_thread_data_wide*> sgd_data_vector_l2;
    unsigned w_wide_len_l2 = N1;
    unsigned array_in_4_l2 = int(w_wide_len_l2 / 4);
    for(unsigned i = 0; i < 4; ++i){
        unsigned start = i * array_in_4_l2;
        unsigned end = start + array_in_4_l2;
        struct SGD_thread_data_wide *sgd_data = (struct SGD_thread_data_wide *)malloc(sizeof(struct SGD_thread_data_wide));
        sgd_data->w_orig = ptr_w2;
        sgd_data->w = ptr_dw_l2;
        sgd_data->LR = lr_;
        sgd_data->length = w_wide_len_l2;
        sgd_data->start = start;
        sgd_data->end = end;
        sgd_data->wide_length = WIDE_LEN;
        pthread_t sgd_thrd;
        sgd_data_vector_l2.push_back(sgd_data);
        sgd_threads_l2.push_back(sgd_thrd);
    }

    std::vector<pthread_t> tanh_threads;
    std::vector<TANH_thread_data_wide*> tanh_data_vector;
    unsigned h1_wide_len = B * (N1 / WIDE_LEN);
    unsigned array_in_4_tanh = int(h1_wide_len / 4);
    for(unsigned i = 0; i < 4; ++i){
        unsigned start = i * array_in_4_tanh;
        unsigned end = start + array_in_4_tanh;
        struct TANH_thread_data_wide *tanh_data = (struct TANH_thread_data_wide *)malloc(sizeof(struct TANH_thread_data_wide));
        tanh_data->in = ptr_h1;
        tanh_data->out = ptr_a1;
        tanh_data->length = h1_wide_len;
        tanh_data->start = start;
        tanh_data->end = end;
        tanh_data->wide_length = WIDE_LEN;
        pthread_t tanh_thrd;
        tanh_data_vector.push_back(tanh_data);
        tanh_threads.push_back(tanh_thrd);
    }

    std::vector<pthread_t> tanh_bw_threads;
    std::vector<TANH_BW_thread_data_wide*> tanh_bw_data_vector;
    h1_wide_len = B * (N1 / WIDE_LEN);
    array_in_4_tanh = int(h1_wide_len / 4);
    for(unsigned i = 0; i < 4; ++i){
        unsigned start = i * array_in_4_tanh;
        unsigned end = start + array_in_4_tanh;
        struct TANH_BW_thread_data_wide *tanh_bw_data = (struct TANH_BW_thread_data_wide *)malloc(sizeof(struct TANH_BW_thread_data_wide));
        tanh_bw_data->in = ptr_a1;
        tanh_bw_data->grad = ptr_dx_l2;
        tanh_bw_data->out = ptr_dx_al2;
        tanh_bw_data->length = h1_wide_len;
        tanh_bw_data->start = start;
        tanh_bw_data->end = end;
        tanh_bw_data->wide_length = WIDE_LEN;
        pthread_t tanh_bw_thrd;
        tanh_bw_data_vector.push_back(tanh_bw_data);
        tanh_bw_threads.push_back(tanh_bw_thrd);
    }


    dB_thread_data bias_args[NUM_THREADS];
    std::vector<HLSNN_DataType> dbgrads[NUM_THREADS];
    setup_db_thread_args(bias_args, dbgrads, ptr_dx_al2, b, n1);

    dB_thread_data bias2_args[NUM_THREADS];
    std::vector<HLSNN_DataType> dbgrads2[NUM_THREADS];
    setup_db_thread_args(bias2_args, dbgrads2, ptr_outgrad, b, WIDE_LEN);

    unsigned num_of_batches = 14;
    unsigned number_of_epochs = 25;

    std::vector<float> training_accuracies;
    std::vector<float> training_losses;

    std::vector<float> epoch_times;
    auto full_training_begin = std::chrono::high_resolution_clock::now();
    for(int ee = 0; ee < number_of_epochs; ee++){
        std::vector<std::vector<wide_type>> inputs_vector;
        std::vector<std::vector<float>> labels_vector;

        for(int bb = 0; bb < num_of_batches; bb++){
            std::string input_path = pathToData + "/epoch_data/epoch_" + std::to_string(ee) + "/" + std::to_string(bb) + ".npz";
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


        auto epoch_start = std::chrono::high_resolution_clock::now();
        float running_acc = 0.0;
        for(int bb = 0; bb < num_of_batches; bb++){

            auto loading_begin = std::chrono::high_resolution_clock::now();
            load_into_buffer(ptr_x, inputs_vector[bb].data(), B * (M/WIDE_LEN));
            //fill_with_zeros_wide(ptr_outgrad, B, WIDE_LEN);
            //fill_with_zeros_wide(ptr_dw_l1, M * (N1/WIDE_LEN), WIDE_LEN);
            //fill_with_zeros_wide(ptr_dw_l2, N1, WIDE_LEN);
            auto loading_end = std::chrono::high_resolution_clock::now();

            auto fpga_fw_layers_begin = std::chrono::high_resolution_clock::now();
            auto fw1_begin = std::chrono::high_resolution_clock::now();
            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_x, buffer_w1, buffer_b1, buffer_h1, buffer_a1, buffer_h2, buffer_w2, buffer_b2, buffer_dw_l1, buffer_dw_l2, buffer_out_grad}, 0 ));
            OCL_CHECK(err, err = q.enqueueTask(krnl_fw1));
            OCL_CHECK(err, err = q.enqueueTask(krnl_tanh));
            OCL_CHECK(err, q.finish()); 
            auto fw1_end = std::chrono::high_resolution_clock::now();
            auto fw2_begin = std::chrono::high_resolution_clock::now();
            OCL_CHECK(err, err = q.enqueueTask(krnl_fw2));
            OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_h2, buffer_h1, buffer_out_grad}, CL_MIGRATE_MEM_OBJECT_HOST));
            OCL_CHECK(err, q.finish());
            auto fw2_end = std::chrono::high_resolution_clock::now();
            auto fpga_fw_layers_end = std::chrono::high_resolution_clock::now();

            auto cpu_softmax_begin = std::chrono::high_resolution_clock::now();
            float acc = softmax_array_2<B, N2>(ptr_h2, ptr_outgrad, labels_vector[bb].data(), b, n2, 1, WIDE_LEN);
            auto cpu_softmax_end = std::chrono::high_resolution_clock::now();

            running_acc += acc;

            auto fpga_bw_begin = std::chrono::high_resolution_clock::now();
            auto dx2_begin = std::chrono::high_resolution_clock::now();
            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_h1, buffer_h2, buffer_out_grad, buffer_dx_l2}, 0 ));
            OCL_CHECK(err, err = q.enqueueTask(krnl_dx2));
            OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_dx_l2}, CL_MIGRATE_MEM_OBJECT_HOST));
            OCL_CHECK(err, q.finish()); 
            auto dx2_end = std::chrono::high_resolution_clock::now();

            auto tanhbw_begin = std::chrono::high_resolution_clock::now();

            for(unsigned i = 0; i < 4; ++i)
            {
                pthread_create(&(tanh_bw_threads[i]), NULL, worker_TANH_BW_wide, (void *)tanh_bw_data_vector[i]);
            }
            
            for(unsigned i = 0; i < 4; ++i)
            {
                pthread_join(tanh_bw_threads[i], NULL);
            }

            auto tanhbw_end = std::chrono::high_resolution_clock::now();

            auto dw2_begin = std::chrono::high_resolution_clock::now();
            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_dx_al2}, 0 ));
            OCL_CHECK(err, err = q.enqueueTask(krnl_dw2));
            OCL_CHECK(err, q.finish()); 
            auto dw2_end = std::chrono::high_resolution_clock::now();

            auto dw1_begin = std::chrono::high_resolution_clock::now();
            OCL_CHECK(err, err = q.enqueueTask(krnl_dw1));
            //OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_x, buffer_w1, buffer_b1, buffer_a1, buffer_h1, buffer_h2, buffer_w2, buffer_dx_l2, buffer_dw_l1, buffer_dw_l2, buffer_b2, buffer_out_grad, buffer_tmp2}, CL_MIGRATE_MEM_OBJECT_HOST));
            OCL_CHECK(err, q.finish());  
            auto dw1_end = std::chrono::high_resolution_clock::now();

            auto fpga_wu2_begin = std::chrono::high_resolution_clock::now();
            //OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_tmp1, buffer_tmp3}, 0 ));
            OCL_CHECK(err, err = q.enqueueTask(krnl_wu2));
            OCL_CHECK(err, q.finish());
            auto fpga_wu2_end = std::chrono::high_resolution_clock::now();

            auto fpga_wu_begin = std::chrono::high_resolution_clock::now();
            //OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_tmp1, buffer_tmp3}, 0 ));
            OCL_CHECK(err, err = q.enqueueTask(krnl_wu));
            OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_x, buffer_dx_al2, buffer_w1, buffer_b1, buffer_a1, buffer_h1, buffer_h2, buffer_w2, buffer_dx_l2, buffer_dw_l1, buffer_dw_l2, buffer_b2, buffer_out_grad}, CL_MIGRATE_MEM_OBJECT_HOST));
            OCL_CHECK(err, q.finish());
            auto fpga_wu_end = std::chrono::high_resolution_clock::now();

            auto fpga_bw_end = std::chrono::high_resolution_clock::now();

            auto cpu_param_update_begin = std::chrono::high_resolution_clock::now();

            auto b2u_begin = std::chrono::high_resolution_clock::now();
            std::vector<HLSNN_DataType> db2 = sumAxis0(ptr_outgrad, b, WIDE_LEN);
            bias_sgd(ptr_b2, db2.data(), WIDE_LEN);
            auto b2u_end = std::chrono::high_resolution_clock::now();

            auto b1u_begin = std::chrono::high_resolution_clock::now();
            std::vector<HLSNN_DataType> db1 = sumAxis0_thread(bias_args);
            bias_sgd(ptr_b1, db1.data(), N1);
            auto b1u_end = std::chrono::high_resolution_clock::now();

            auto cpu_param_update_end = std::chrono::high_resolution_clock::now();

            #ifdef PRINT_DEBUG
            if(ee == 0 && bb == 0){
                out_file << "========================================================" << std::endl;
                out_file << "H1:" << std::endl;
                print_output_wide(ptr_h1, B, N1, WIDE_LEN, out_file);
                out_file << "========================================================" << std::endl;
                out_file << "H2:" << std::endl;
                print_output_wide(ptr_h2, B, N2, WIDE_LEN, out_file);
                out_file << "========================================================" << std::endl;
                out_file << "========================================================" << std::endl;
                out_file << "OGRAD:" << std::endl;
                print_output_wide(ptr_outgrad, B, WIDE_LEN, WIDE_LEN, out_file);
                out_file << "========================================================" << std::endl;
                out_file << "B2 UPDATED:" << std::endl;
                print_output(ptr_b2, 1, WIDE_LEN, out_file);
                out_file << "========================================================" << std::endl;
                out_file << "DH1:" << std::endl;
                print_output_wide(ptr_dx_l2, B, N1, WIDE_LEN, out_file);
                out_file << "========================================================" << std::endl;
                out_file << "B1 UPDATED:" << std::endl;
                print_output(ptr_b1, 1, N1, out_file);
                out_file << "========================================================" << std::endl;
                out_file << "========================================================" << std::endl;
                out_file << "dW2:" << std::endl;
                print_output_wide(ptr_w2, N1, WIDE_LEN, WIDE_LEN, out_file);
                out_file << "========================================================" << std::endl;
                out_file << "========================================================" << std::endl;
                out_file << "dW1:" << std::endl;
                print_output_wide(ptr_w1, M, N1, WIDE_LEN, out_file);
                out_file << "========================================================" << std::endl;
            }
            #endif

            #ifdef MEASURE_TIME
            std::chrono::duration<double> fw_time = fpga_fw_layers_end - fpga_fw_layers_begin;
            std::chrono::duration<double> bw_time = fpga_bw_end - fpga_bw_begin;
            std::chrono::duration<double> softmax_time = cpu_softmax_end - cpu_softmax_begin;
            std::chrono::duration<double> param_update = cpu_param_update_end - cpu_param_update_begin;
            printf("- FPGA Time FW         : %10.4f ms\n",    fw_time.count() * 1000.0);
            printf("- FPGA Time BW         : %10.4f ms\n",    bw_time.count() * 1000.0);
            printf("- CPU Softmax Time     : %10.4f ms\n",    softmax_time.count() * 1000.0);
            printf("- CPU Update Time      : %10.4f ms\n",    param_update.count() * 1000.0);


            std::chrono::duration<double> fw1 = fw1_end - fw1_begin;
            std::chrono::duration<double> fw2 = fw2_end - fw2_begin;
            std::chrono::duration<double> dx2 = dx2_end - dx2_begin;
            std::chrono::duration<double> dw2 = dw2_end - dw2_begin;
            std::chrono::duration<double> dw1 = dw1_end - dw1_begin;
            printf("- FW1 Time      : %10.4f ms\n",    fw1.count() * 1000.0);
            printf("- FW2 Time      : %10.4f ms\n",    fw2.count() * 1000.0);
            printf("- DX2 Time      : %10.4f ms\n",    dx2.count() * 1000.0);
            printf("- DW2 Time      : %10.4f ms\n",    dw2.count() * 1000.0);
            printf("- DW1 Time      : %10.4f ms\n",    dw1.count() * 1000.0);

            std::chrono::duration<double> b1_update = b1u_end - b1u_begin;
            std::chrono::duration<double> b2_update = b2u_end - b2u_begin;
            printf("- bU2 Update Time      : %10.4f ms\n",    b2_update.count() * 1000.0);
            printf("- bU1 Update Time      : %10.4f ms\n",    b1_update.count() * 1000.0);

            std::chrono::duration<double> wu_duration = fpga_wu_end - fpga_wu_begin;
            std::chrono::duration<double> wu2_duration = fpga_wu2_end - fpga_wu2_begin;
            printf("- FPGA WU1 Time      : %10.4f ms\n",    wu_duration.count() * 1000.0);
            printf("- FPGA WU2 Time      : %10.4f ms\n",    wu2_duration.count() * 1000.0);

            std::chrono::duration<double> tanhbw_duration = tanhbw_end - tanhbw_begin;
            printf("- CPU tanhbw Time      : %10.4f ms\n",    tanhbw_duration.count() * 1000.0);

            std::chrono::duration<double> setup_duration = loading_end - loading_begin;
            printf("- CPU setup Time      : %10.4f ms\n",    setup_duration.count() * 1000.0);



            //printf("- bU1 Update Time      : %10.4f s\n",    b1_update.count() * 1000.0);
            printf("------------------\n");
            #endif



            #ifdef RUN_COMPARE
            
            auto wu1_thread_begin = std::chrono::high_resolution_clock::now();
            // SGD multi-threading
            // I double checked this for multithreading, it does update all elements
            for(unsigned i = 0; i < threading_factor; ++i)
            {
                pthread_create(&(sgd_threads[i]), NULL, worker_SGD_wide, (void *)sgd_data_vector[i]);
            }
            
            for(unsigned i = 0; i < threading_factor; ++i)
            {
                pthread_join(sgd_threads[i], NULL);
            }
            auto wu1_thread_end = std::chrono::high_resolution_clock::now();

            auto w2u_begin = std::chrono::high_resolution_clock::now();
            for(unsigned i = 0; i < 4; ++i)
            {
                pthread_create(&(sgd_threads_l2[i]), NULL, worker_SGD_wide, (void *)sgd_data_vector_l2[i]);
            }
            
            for(unsigned i = 0; i < 4; ++i)
            {
                pthread_join(sgd_threads_l2[i], NULL);
            }
            auto w2u_end = std::chrono::high_resolution_clock::now();

            auto wu1_single_begin = std::chrono::high_resolution_clock::now();
            weight_sgd(ptr_w1, ptr_dw_l1, M, N1, WIDE_LEN);
            auto wu1_single_end = std::chrono::high_resolution_clock::now();

            auto wu2_single_begin = std::chrono::high_resolution_clock::now();
            weight_sgd(ptr_w2, ptr_dw_l2, N1, WIDE_LEN, WIDE_LEN);
            auto wu2_single_end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> w1_update_thread = wu1_thread_end - wu1_thread_begin;
            std::chrono::duration<double> w1_update_single = wu1_single_end - wu1_single_begin;
            std::chrono::duration<double> w2_update_single = wu2_single_end - wu2_single_begin;
            std::chrono::duration<double> w2_update = w2u_end - w2u_begin;

            printf("- WU1 (4 threads) Update Time      : %10.4f ms\n",    w1_update_thread.count() * 1000.0);
            printf("- WU1 (1 thread) Update Time      : %10.4f ms\n",    w1_update_single.count() * 1000.0);

            printf("- WU2 (4 threads) Update Time      : %10.4f ms\n",    w2_update.count() * 1000.0);
            printf("- WU2 (1 thread) Update Time      : %10.4f ms\n",    w2_update_single.count() * 1000.0);

            auto tanhbw_single_begin = std::chrono::high_resolution_clock::now();
            tanh_activation_bw(ptr_dx_al2, ptr_a1, ptr_dx_l2, B, N1, WIDE_LEN);
            auto tanhbw_single_end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> tanhbw_single_duration = tanhbw_single_end - tanhbw_single_begin;
            printf("- Tanh BW (4 threads) Time      : %10.4f ms\n",    tanhbw_duration.count() * 1000.0);
            printf("- Tanh BW (1 thread) Time      : %10.4f ms\n",    tanhbw_single_duration.count() * 1000.0);

            auto b1u_single_begin = std::chrono::high_resolution_clock::now();
            //std::vector<HLSNN_DataType> db1_single = sumAxis0(ptr_dx_al2, b, n1);
            std::vector<HLSNN_DataType> db1_single = sumAxis0(ptr_dx_al2, b, n1, WIDE_LEN);
            bias_sgd(ptr_b1, db1_single.data(), N1);
            auto b1u_single_end = std::chrono::high_resolution_clock::now();

            auto b2u_thread_begin = std::chrono::high_resolution_clock::now();
            std::vector<HLSNN_DataType> db2_thread = sumAxis0_thread(bias2_args);
            bias_sgd(ptr_b2, db2_thread.data(), WIDE_LEN);
            auto b2u_thread_end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> db1_single_duration = b1u_single_end - b1u_single_begin;
            std::chrono::duration<double> db2_thread_duration = b2u_thread_end - b2u_thread_begin;
            printf("- bU1 (4 threads) Update Time      : %10.4f ms\n",   b1_update.count() * 1000.0);
            printf("- bU1 (1 thread) Update Time      : %10.4f ms\n",    db1_single_duration.count() * 1000.0);
            printf("- bU2 (4 threads) Update Time      : %10.4f ms\n",   db2_thread_duration.count() * 1000.0);
            printf("- bU2 (1 thread) Update Time      : %10.4f ms\n",    b2_update.count() * 1000.0);

            #endif
        }
        auto epoch_end = std::chrono::high_resolution_clock::now();

        running_acc = (running_acc / float(num_of_batches * 32)) * 100.0;
        std::cout << "Accuracy is: " << running_acc << std::endl;

        std::chrono::duration<double> epoch_runtime = epoch_end - epoch_start;
        printf("- Epoch runtime         : %10.4f s\n",    epoch_runtime.count());

        training_accuracies.push_back(running_acc);

        epoch_times.push_back(epoch_runtime.count());
        
    }
    auto full_training_end = std::chrono::high_resolution_clock::now();

    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_x, ptr_x));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_w1, ptr_w1));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_w2, ptr_w2));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_b1, ptr_b1));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_b2, ptr_b2));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_a1, ptr_a1));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_h1, ptr_h1));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_h2, ptr_h2));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dx_l2, ptr_dx_l2));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dx_al2, ptr_dx_al2));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dw_l2, ptr_dw_l2));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dw_l1, ptr_dw_l1));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_out_grad, ptr_outgrad));
    OCL_CHECK(err, err = q.finish());

    std::cout << "Finished running program" << std::endl;

    std::cout << "\n TRAINING ACCURACIES \n" << std::endl;
    for(const float& i : training_accuracies){
        std::cout << std::to_string(i) << ",";
    }
    std::cout << std::endl;

    float epoch_sum = 0.0;
    for(const float& i : epoch_times){
        epoch_sum += i;
    }
    std::cout << "Total runtime: " << epoch_sum << std::endl;

    std::chrono::duration<double> full_runtime = full_training_end - full_training_begin;
    printf("Total runtime (with loading): %10.4f s\n", full_runtime.count());
    

    std::cout << "Finished running!" << std::endl;

    return 0;
}