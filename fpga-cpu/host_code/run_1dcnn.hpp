#include <CL/cl.h>
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

//#include "vadd.h"
#include <CL/cl2.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <chrono>
#include <pthread.h>
#include <string> 
#include "cnpy.h"

#include "ap_fixed.h"
#include "hls_vector.h"
#include "hls_math.h"
#include "types.hpp"


#define MEASURE_TIME
#define PRINT_DEBUG

/////////////////////////////////////////////////

#define CNN1D_X_LENGTH 128

//CONV1 + MP
#define CNN1D_CONV_1_C 6
#define CNN1D_CONV_1_K 2
#define CNN1D_CONV_1_F 32
#define CNN1D_CONV_1_C_im2col (6 * 2)
#define CNN1D_CONV_1_C_im2col_padded 16
#define CNN1D_CONV_1_W 128
#define CNN1D_CONV_1_MP_W 64
#define CNN1D_CONV_1_W_conv 64

//CONV2 + MP
#define CNN1D_CONV_2_C 32
#define CNN1D_CONV_2_K 2
#define CNN1D_CONV_2_F 16
#define CNN1D_CONV_2_C_im2col (32 * 2)
#define CNN1D_CONV_2_W 64
#define CNN1D_CONV_2_MP_W 32
#define CNN1D_CONV_2_W_conv 32

//CONV3
#define CNN1D_CONV_3_C 16
#define CNN1D_CONV_3_K 2
#define CNN1D_CONV_3_F 16
#define CNN1D_CONV_3_C_im2col (16 * 2)
#define CNN1D_CONV_3_W 32
#define CNN1D_CONV_3_W_conv 32

//CONV4
#define CNN1D_CONV_4_C 16
#define CNN1D_CONV_4_K 2
#define CNN1D_CONV_4_F 16
#define CNN1D_CONV_4_C_im2col (16 * 2)
#define CNN1D_CONV_4_W 32
#define CNN1D_CONV_4_W_conv 32

//CONV5
#define CNN1D_CONV_5_C 16
#define CNN1D_CONV_5_K 2
#define CNN1D_CONV_5_F 16
#define CNN1D_CONV_5_C_im2col (16 * 2)
#define CNN1D_CONV_5_W 32
#define CNN1D_CONV_5_W_conv 32

#define CNN1D_FC_1_C 512
#define CNN1D_FC_1_F 3
#define CNN1D_FC_1_F_widened 16

#define MAX_ARRAY_SIZE 3000

// This function does ONE inference and backward pass on a 1D-CNN network and measures the latency for individual computations.
int run_1dcnn_train(std::string xclbinFilename_, std::string pathToData_, cl::Context &context, cl::CommandQueue &q, cl::Program &program) {

    std::string xclbinFilename = xclbinFilename_;
    std::string pathToData = pathToData_;

    std::ofstream out_file;
    out_file.open ("./output.txt");

    std::vector<cl::Device> devices;
    cl_int err;

    cl::Kernel krnl_conv1_fw, krnl_conv2_fw, krnl_conv3_fw, krnl_conv4_fw, krnl_conv5_fw, krnl_fc1_fw;
    cl::Kernel krnl_fc1_dx, krnl_conv5_dx, krnl_conv4_dx, krnl_conv3_dx, krnl_conv2_dx, krnl_conv1_dx;
    cl::Kernel krnl_fc1_dw, krnl_conv1_dw, krnl_conv2_dw, krnl_conv3_dw, krnl_conv4_dw, krnl_conv5_dw;
    cl::Kernel krnl_wu_fc1, krnl_wu_conv1, krnl_wu_conv2, krnl_wu_conv3, krnl_wu_conv4, krnl_wu_conv5;

    OCL_CHECK(err, krnl_conv1_fw = cl::Kernel(program, "top_mm_im2col", &err));
    OCL_CHECK(err, krnl_conv2_fw = cl::Kernel(program, "top_mm_im2col", &err));
    OCL_CHECK(err, krnl_conv3_fw = cl::Kernel(program, "top_mm_im2col", &err));
    OCL_CHECK(err, krnl_conv4_fw = cl::Kernel(program, "top_mm_im2col", &err));
    OCL_CHECK(err, krnl_conv5_fw = cl::Kernel(program, "top_mm_im2col", &err));
    OCL_CHECK(err, krnl_fc1_fw = cl::Kernel(program, "top_mm_im2col", &err));

    OCL_CHECK(err, krnl_fc1_dx = cl::Kernel(program, "top_mm_im2col", &err));
    OCL_CHECK(err, krnl_conv5_dx = cl::Kernel(program, "top_mm_im2col", &err));
    OCL_CHECK(err, krnl_conv4_dx = cl::Kernel(program, "top_mm_im2col", &err));
    OCL_CHECK(err, krnl_conv3_dx = cl::Kernel(program, "top_mm_im2col", &err));
    OCL_CHECK(err, krnl_conv2_dx = cl::Kernel(program, "top_mm_im2col", &err));
    OCL_CHECK(err, krnl_conv1_dx = cl::Kernel(program, "top_mm_im2col", &err));

    OCL_CHECK(err, krnl_fc1_dw = cl::Kernel(program, "top_mm_im2col", &err));
    OCL_CHECK(err, krnl_conv1_dw = cl::Kernel(program, "top_mm_im2col", &err));
    OCL_CHECK(err, krnl_conv2_dw = cl::Kernel(program, "top_mm_im2col", &err));
    OCL_CHECK(err, krnl_conv3_dw = cl::Kernel(program, "top_mm_im2col", &err));
    OCL_CHECK(err, krnl_conv4_dw = cl::Kernel(program, "top_mm_im2col", &err));
    OCL_CHECK(err, krnl_conv5_dw = cl::Kernel(program, "top_mm_im2col", &err));


    OCL_CHECK(err, krnl_wu_fc1 = cl::Kernel(program, "top_wu", &err));
    OCL_CHECK(err, krnl_wu_conv1 = cl::Kernel(program, "top_wu", &err));
    OCL_CHECK(err, krnl_wu_conv2 = cl::Kernel(program, "top_wu", &err));
    OCL_CHECK(err, krnl_wu_conv3 = cl::Kernel(program, "top_wu", &err));
    OCL_CHECK(err, krnl_wu_conv4 = cl::Kernel(program, "top_wu", &err));
    OCL_CHECK(err, krnl_wu_conv5 = cl::Kernel(program, "top_wu", &err));


    ////================================================

    std::string weights_path = pathToData + "/im2col_1dcnn_test.npz";
	std::vector<std::string> layer_names = {"x", "y", "x_orig", "w1", "w1_tr", "w2", "w2_tr", "w3", "w3_tr", "w4", "w4_tr", "w5", "w5_tr", "b1", "b2", "b3", "b4", "b5", "fcw1", "fcw1_padded", "fcb1"};
	std::map<std::string, std::vector<HLSNN_DataType>> dataMap;
	std::map<std::string, std::vector<double>> dataMap_float;

    readNpz(weights_path, layer_names, dataMap_float);

    std::map<std::string, std::vector<double>>::iterator it;
    for (it = dataMap_float.begin(); it != dataMap_float.end(); it++)
    {
    	std::vector<double> tmp = it->second;
    	std::vector<HLSNN_DataType> tmp_result;

    	std::cout << tmp.size() << std::endl;

        for(int i = 0; i < tmp.size(); i++){
        	//tmp_result.push_back(HLSNN_DataType(tmp[i]));
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

    // Pointers
    HLSNN_DataType* x_ptr = dataMap["x"].data();    
    HLSNN_DataType* conv1_w_ptr = dataMap["w1"].data();
    HLSNN_DataType* conv2_w_ptr = dataMap["w2"].data();
    HLSNN_DataType* conv3_w_ptr = dataMap["w3"].data();
    HLSNN_DataType* conv4_w_ptr = dataMap["w4"].data();
    HLSNN_DataType* conv5_w_ptr = dataMap["w5"].data();
    HLSNN_DataType* conv1_b_ptr = dataMap["b1"].data();
    HLSNN_DataType* conv2_b_ptr = dataMap["b2"].data();
    HLSNN_DataType* conv3_b_ptr = dataMap["b3"].data();
    HLSNN_DataType* conv4_b_ptr = dataMap["b4"].data();
    HLSNN_DataType* conv5_b_ptr = dataMap["b5"].data();

    HLSNN_DataType* fcw1_w_ptr = dataMap["fcw1"].data();
    HLSNN_DataType* fcw1_w_padded_ptr = dataMap["fcw1_padded"].data();
    HLSNN_DataType* fcw1_b_ptr = dataMap["fcb1"].data();

    HLSNN_DataType* y_ptr = dataMap["y"].data();

    std::cout << "All data read in" << std::endl;

    ////============================================================================

    //Input storage buffer
    // Buffers and pointers to the buffers are defined for data storage. These storages are accessible to both FPGA and CPU.
    wide_type *ptr_input;
    unsigned input_size = CNN1D_CONV_1_C_im2col_padded * CNN1D_CONV_1_W;
    OCL_CHECK(err, cl::Buffer buffer_input(context, CL_MEM_READ_ONLY, (input_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_input = (wide_type*)q.enqueueMapBuffer(buffer_input, CL_TRUE, CL_MAP_WRITE, 0, (input_size) * sizeof(wide_type), NULL, NULL, &err));

    //Label storage
    std::vector<HLSNN_DataType> label;

    //Weight storage declarations
    // Buffers and pointers to the buffers are defined for data storage. These storages are accessible to both FPGA and CPU.
    unsigned conv1_size = CNN1D_CONV_1_F * CNN1D_CONV_1_C_im2col_padded * CNN1D_CONV_1_K;
    unsigned conv2_size = CNN1D_CONV_2_F * CNN1D_CONV_2_C_im2col * CNN1D_CONV_2_K;
    unsigned conv3_size = CNN1D_CONV_3_F * CNN1D_CONV_3_C_im2col * CNN1D_CONV_3_K;
    unsigned conv4_size = CNN1D_CONV_4_F * CNN1D_CONV_4_C_im2col * CNN1D_CONV_4_K;
    unsigned conv5_size = CNN1D_CONV_5_F * CNN1D_CONV_5_C_im2col * CNN1D_CONV_5_K;
    unsigned conv1b_size = CNN1D_CONV_1_F;
    unsigned conv2b_size = CNN1D_CONV_2_F;
    unsigned conv3b_size = CNN1D_CONV_3_F;
    unsigned conv4b_size = CNN1D_CONV_4_F;
    unsigned conv5b_size = CNN1D_CONV_5_F;
    wide_type *ptr_conv1w, *ptr_conv2w, *ptr_conv3w, *ptr_conv4w, *ptr_conv5w, *ptr_conv1w_tr, *ptr_conv2w_tr, *ptr_conv3w_tr, *ptr_conv4w_tr, *ptr_conv5w_tr;
    HLSNN_DataType *ptr_conv1b, *ptr_conv2b, *ptr_conv3b, *ptr_conv4b, *ptr_conv5b;
    OCL_CHECK(err, cl::Buffer buffer_conv1w(context, CL_MEM_READ_ONLY, (conv1_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_conv1w = (wide_type*)q.enqueueMapBuffer(buffer_conv1w, CL_TRUE, CL_MAP_WRITE, 0, (conv1_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_conv2w(context, CL_MEM_READ_ONLY, (conv2_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_conv2w = (wide_type*)q.enqueueMapBuffer(buffer_conv2w, CL_TRUE, CL_MAP_WRITE, 0, (conv2_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_conv3w(context, CL_MEM_READ_ONLY, (conv3_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_conv3w = (wide_type*)q.enqueueMapBuffer(buffer_conv3w, CL_TRUE, CL_MAP_WRITE, 0, (conv3_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_conv4w(context, CL_MEM_READ_ONLY, (conv4_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_conv4w = (wide_type*)q.enqueueMapBuffer(buffer_conv4w, CL_TRUE, CL_MAP_WRITE, 0, (conv4_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_conv5w(context, CL_MEM_READ_ONLY, (conv5_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_conv5w = (wide_type*)q.enqueueMapBuffer(buffer_conv5w, CL_TRUE, CL_MAP_WRITE, 0, (conv5_size) * sizeof(wide_type), NULL, NULL, &err));

    OCL_CHECK(err, cl::Buffer buffer_conv1w_tr(context, CL_MEM_READ_ONLY, (conv1_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_conv1w_tr = (wide_type*)q.enqueueMapBuffer(buffer_conv1w_tr, CL_TRUE, CL_MAP_WRITE, 0, (conv1_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_conv2w_tr(context, CL_MEM_READ_ONLY, (conv2_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_conv2w_tr = (wide_type*)q.enqueueMapBuffer(buffer_conv2w_tr, CL_TRUE, CL_MAP_WRITE, 0, (conv2_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_conv3w_tr(context, CL_MEM_READ_ONLY, (conv3_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_conv3w_tr = (wide_type*)q.enqueueMapBuffer(buffer_conv3w_tr, CL_TRUE, CL_MAP_WRITE, 0, (conv3_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_conv4w_tr(context, CL_MEM_READ_ONLY, (conv4_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_conv4w_tr = (wide_type*)q.enqueueMapBuffer(buffer_conv4w_tr, CL_TRUE, CL_MAP_WRITE, 0, (conv4_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_conv5w_tr(context, CL_MEM_READ_ONLY, (conv5_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_conv5w_tr = (wide_type*)q.enqueueMapBuffer(buffer_conv5w_tr, CL_TRUE, CL_MAP_WRITE, 0, (conv5_size) * sizeof(wide_type), NULL, NULL, &err));
    
    OCL_CHECK(err, cl::Buffer buffer_conv1b(context, CL_MEM_READ_ONLY, (conv1b_size) * sizeof(HLSNN_DataType), NULL, &err));
    OCL_CHECK(err, ptr_conv1b = (HLSNN_DataType*)q.enqueueMapBuffer(buffer_conv1b, CL_TRUE, CL_MAP_WRITE, 0, (conv1b_size) * sizeof(HLSNN_DataType), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_conv2b(context, CL_MEM_READ_ONLY, (conv2b_size) * sizeof(HLSNN_DataType), NULL, &err));
    OCL_CHECK(err, ptr_conv2b = (HLSNN_DataType*)q.enqueueMapBuffer(buffer_conv2b, CL_TRUE, CL_MAP_WRITE, 0, (conv2b_size) * sizeof(HLSNN_DataType), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_conv3b(context, CL_MEM_READ_ONLY, (conv3b_size) * sizeof(HLSNN_DataType), NULL, &err));
    OCL_CHECK(err, ptr_conv3b = (HLSNN_DataType*)q.enqueueMapBuffer(buffer_conv3b, CL_TRUE, CL_MAP_WRITE, 0, (conv3b_size) * sizeof(HLSNN_DataType), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_conv4b(context, CL_MEM_READ_ONLY, (conv4b_size) * sizeof(HLSNN_DataType), NULL, &err));
    OCL_CHECK(err, ptr_conv4b = (HLSNN_DataType*)q.enqueueMapBuffer(buffer_conv4b, CL_TRUE, CL_MAP_WRITE, 0, (conv4b_size) * sizeof(HLSNN_DataType), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_conv5b(context, CL_MEM_READ_ONLY, (conv5b_size) * sizeof(HLSNN_DataType), NULL, &err));
    OCL_CHECK(err, ptr_conv5b = (HLSNN_DataType*)q.enqueueMapBuffer(buffer_conv5b, CL_TRUE, CL_MAP_WRITE, 0, (conv5b_size) * sizeof(HLSNN_DataType), NULL, NULL, &err));

    unsigned fc1w_size = CNN1D_FC_1_C * CNN1D_FC_1_F_widened;
    unsigned fc1b_size = CNN1D_FC_1_F_widened;
    wide_type *ptr_fc1w;
    HLSNN_DataType *ptr_fc1b;
    OCL_CHECK(err, cl::Buffer buffer_fc1w(context, CL_MEM_READ_ONLY, (fc1w_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_fc1w = (wide_type*)q.enqueueMapBuffer(buffer_fc1w, CL_TRUE, CL_MAP_WRITE, 0, (fc1w_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_fc1b(context, CL_MEM_READ_ONLY, (fc1b_size) * sizeof(HLSNN_DataType), NULL, &err));
    OCL_CHECK(err, ptr_fc1b = (HLSNN_DataType*)q.enqueueMapBuffer(buffer_fc1b, CL_TRUE, CL_MAP_WRITE, 0, (fc1b_size) * sizeof(HLSNN_DataType), NULL, NULL, &err));

    //CONV 1
	place_in_wide(conv1_w_ptr, ptr_conv1w, CNN1D_CONV_1_F, CNN1D_CONV_1_C_im2col_padded, WIDE_LEN); //CONV 1 weight
	place_in_vector(ptr_conv1b, conv1_b_ptr, CNN1D_CONV_1_F); //CONV 1 bias
    transpose_matrix_widened(ptr_conv1w, ptr_conv1w_tr, CNN1D_CONV_1_F, CNN1D_CONV_1_C_im2col_padded, WIDE_LEN);

    //CONV2
	place_in_wide(conv2_w_ptr, ptr_conv2w, CNN1D_CONV_2_F, CNN1D_CONV_2_C_im2col, WIDE_LEN); //CONV 2 weight
    place_in_vector(ptr_conv2b, conv2_b_ptr, CNN1D_CONV_2_F); //CONV 2 bias
    transpose_matrix_widened(ptr_conv2w, ptr_conv2w_tr, CNN1D_CONV_2_F, CNN1D_CONV_2_C_im2col, WIDE_LEN);

    //CONV3
	place_in_wide(conv3_w_ptr, ptr_conv3w, CNN1D_CONV_3_F, CNN1D_CONV_3_C_im2col, WIDE_LEN); //CONV 3 weight
    place_in_vector(ptr_conv3b, conv3_b_ptr, CNN1D_CONV_3_F); //CONV 3 bias
    transpose_matrix_widened(ptr_conv3w, ptr_conv3w_tr, CNN1D_CONV_3_F, CNN1D_CONV_3_C_im2col, WIDE_LEN);

    //CONV4
	place_in_wide(conv4_w_ptr, ptr_conv4w, CNN1D_CONV_4_F, CNN1D_CONV_4_C_im2col, WIDE_LEN); //CONV 4 weight
    place_in_vector(ptr_conv4b, conv4_b_ptr, CNN1D_CONV_4_F); //CONV 4 bias
    transpose_matrix_widened(ptr_conv4w, ptr_conv4w_tr, CNN1D_CONV_4_F, CNN1D_CONV_4_C_im2col, WIDE_LEN);

    //CONV5
	place_in_wide(conv5_w_ptr, ptr_conv5w, CNN1D_CONV_5_F, CNN1D_CONV_5_C_im2col, WIDE_LEN); //CONV 5 weight
    place_in_vector(ptr_conv5b, conv5_b_ptr, CNN1D_CONV_5_F); //CONV 5 bias
    transpose_matrix_widened(ptr_conv5w, ptr_conv5w_tr, CNN1D_CONV_5_F, CNN1D_CONV_5_C_im2col, WIDE_LEN);


    place_in_wide_widen(fcw1_w_padded_ptr, ptr_fc1w, 512, CNN1D_FC_1_F, CNN1D_FC_1_F_widened, WIDE_LEN); //FC 1 weight padded
    place_in_vector_pad(ptr_fc1b, fcw1_b_ptr, CNN1D_FC_1_F_widened, CNN1D_FC_1_F); //FC 1 bias


    ////============================================================================


    out_file << std::endl << "################################## INITIAL DATA ##################################" << std::endl;

	out_file << "W1 padded:" << std::endl;
	print_output_wide(ptr_conv1w, CNN1D_CONV_1_F, CNN1D_CONV_1_C_im2col_padded, WIDE_LEN, out_file);

	out_file << "b1:" << std::endl;
    print_vector(ptr_conv1b, CNN1D_CONV_1_F, out_file);

	out_file << "W2 weight:" << std::endl;
	print_output_wide(ptr_conv2w, CNN1D_CONV_2_F, CNN1D_CONV_2_C_im2col, WIDE_LEN, out_file);

	out_file << "b2:" << std::endl;
	print_vector(ptr_conv2b, CNN1D_CONV_2_F, out_file);

	out_file << "W3 weight:" << std::endl;
	print_output_wide(ptr_conv3w, CNN1D_CONV_3_F, CNN1D_CONV_3_C_im2col, WIDE_LEN, out_file);

	out_file << "W3 weight transposed:" << std::endl;
	print_output_wide(ptr_conv3w_tr, CNN1D_CONV_3_C_im2col, CNN1D_CONV_3_F, WIDE_LEN, out_file);

	out_file << "b3:" << std::endl;
	print_vector(ptr_conv3b, CNN1D_CONV_3_F, out_file);

	out_file << "W4 weight:" << std::endl;
	print_output_wide(ptr_conv4w, CNN1D_CONV_4_F, CNN1D_CONV_4_C_im2col, WIDE_LEN, out_file);

	out_file << "W4 weight transposed:" << std::endl;
	print_output_wide(ptr_conv4w_tr, CNN1D_CONV_4_C_im2col, CNN1D_CONV_4_F, WIDE_LEN, out_file);

	out_file << "b4:" << std::endl;
	print_vector(ptr_conv4b, CNN1D_CONV_4_F, out_file);

	out_file << "W5 weight:" << std::endl;
	print_output_wide(ptr_conv5w, CNN1D_CONV_5_F, CNN1D_CONV_5_C_im2col, WIDE_LEN, out_file);

	out_file << "W5 weight transposed:" << std::endl;
	print_output_wide(ptr_conv5w_tr, CNN1D_CONV_5_C_im2col, CNN1D_CONV_5_F, WIDE_LEN, out_file);

	out_file << "b5:" << std::endl;
	print_vector(ptr_conv5b, CNN1D_CONV_5_F, out_file);

	out_file << "FC1 weight padded:" << std::endl;
	print_output_wide(ptr_fc1w, CNN1D_FC_1_C, CNN1D_FC_1_F_widened, WIDE_LEN, out_file);

	out_file << "FC1 bias:" << std::endl;
	print_vector(ptr_fc1b, CNN1D_FC_1_F_widened, out_file);


    ////============================================================================
    // Output storage declarations
    // Buffers and pointers to the buffers are defined for data storage. These storages are accessible to both FPGA and CPU.

    unsigned *ptr_conv1_ix, *ptr_conv2_ix, *ptr_conv3_ix, *ptr_conv4_ix, *ptr_conv5_ix, *ptr_fc1_ix;
    OCL_CHECK(err, cl::Buffer buffer_conv1_ix(context, CL_MEM_READ_WRITE, (MAX_ARRAY_SIZE) * sizeof(unsigned), NULL, &err));
    OCL_CHECK(err, ptr_conv1_ix = (unsigned*)q.enqueueMapBuffer(buffer_conv1_ix, CL_TRUE, CL_MAP_WRITE, 0, (MAX_ARRAY_SIZE) * sizeof(unsigned), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_conv2_ix(context, CL_MEM_READ_WRITE, (MAX_ARRAY_SIZE) * sizeof(unsigned), NULL, &err));
    OCL_CHECK(err, ptr_conv2_ix = (unsigned*)q.enqueueMapBuffer(buffer_conv2_ix, CL_TRUE, CL_MAP_WRITE, 0, (MAX_ARRAY_SIZE) * sizeof(unsigned), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_conv3_ix(context, CL_MEM_READ_WRITE, (MAX_ARRAY_SIZE) * sizeof(unsigned), NULL, &err));
    OCL_CHECK(err, ptr_conv3_ix = (unsigned*)q.enqueueMapBuffer(buffer_conv3_ix, CL_TRUE, CL_MAP_WRITE, 0, (MAX_ARRAY_SIZE) * sizeof(unsigned), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_conv4_ix(context, CL_MEM_READ_WRITE, (MAX_ARRAY_SIZE) * sizeof(unsigned), NULL, &err));
    OCL_CHECK(err, ptr_conv4_ix = (unsigned*)q.enqueueMapBuffer(buffer_conv4_ix, CL_TRUE, CL_MAP_WRITE, 0, (MAX_ARRAY_SIZE) * sizeof(unsigned), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_conv5_ix(context, CL_MEM_READ_WRITE, (MAX_ARRAY_SIZE) * sizeof(unsigned), NULL, &err));
    OCL_CHECK(err, ptr_conv5_ix = (unsigned*)q.enqueueMapBuffer(buffer_conv5_ix, CL_TRUE, CL_MAP_WRITE, 0, (MAX_ARRAY_SIZE) * sizeof(unsigned), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_fc1_ix(context, CL_MEM_READ_WRITE, (MAX_ARRAY_SIZE) * sizeof(unsigned), NULL, &err));
    OCL_CHECK(err, ptr_fc1_ix = (unsigned*)q.enqueueMapBuffer(buffer_fc1_ix, CL_TRUE, CL_MAP_WRITE, 0, (MAX_ARRAY_SIZE) * sizeof(unsigned), NULL, NULL, &err));
    
    unsigned mp1_size = CNN1D_CONV_1_F * CNN1D_CONV_1_K * CNN1D_CONV_1_W;
    wide_type *ptr_mp1, *ptr_dmp1, *ptr_dconv1;
    OCL_CHECK(err, cl::Buffer buffer_mp1(context, CL_MEM_READ_WRITE, (mp1_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_mp1 = (wide_type*)q.enqueueMapBuffer(buffer_mp1, CL_TRUE, CL_MAP_READ, 0, (mp1_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dmp1(context, CL_MEM_READ_WRITE, (mp1_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_dmp1 = (wide_type*)q.enqueueMapBuffer(buffer_dmp1, CL_TRUE, CL_MAP_READ, 0, (mp1_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dconv1(context, CL_MEM_READ_WRITE, (mp1_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_dconv1 = (wide_type*)q.enqueueMapBuffer(buffer_dconv1, CL_TRUE, CL_MAP_READ, 0, (mp1_size) * sizeof(wide_type), NULL, NULL, &err));
        
    unsigned mp2_size = CNN1D_CONV_2_F * CNN1D_CONV_2_K * CNN1D_CONV_2_W;
    wide_type *ptr_mp2, *ptr_dmp2, *ptr_dconv2;
    OCL_CHECK(err, cl::Buffer buffer_mp2(context, CL_MEM_READ_WRITE, (mp2_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_mp2 = (wide_type*)q.enqueueMapBuffer(buffer_mp2, CL_TRUE, CL_MAP_READ, 0, (mp2_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dmp2(context, CL_MEM_READ_WRITE, (mp2_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_dmp2 = (wide_type*)q.enqueueMapBuffer(buffer_dmp2, CL_TRUE, CL_MAP_READ, 0, (mp2_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dconv2(context, CL_MEM_READ_WRITE, (mp2_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_dconv2 = (wide_type*)q.enqueueMapBuffer(buffer_dconv2, CL_TRUE, CL_MAP_READ, 0, (mp2_size) * sizeof(wide_type), NULL, NULL, &err));

    wide_type *ptr_conv3, *ptr_dconv3, *ptr_dconv3_relu;
    OCL_CHECK(err, cl::Buffer buffer_conv3(context, CL_MEM_READ_WRITE, (conv3_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_conv3 = (wide_type*)q.enqueueMapBuffer(buffer_conv3, CL_TRUE, CL_MAP_READ, 0, (conv3_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dconv3(context, CL_MEM_READ_WRITE, (conv3_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_dconv3 = (wide_type*)q.enqueueMapBuffer(buffer_dconv3, CL_TRUE, CL_MAP_READ, 0, (conv3_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dconv3_relu(context, CL_MEM_READ_WRITE, (conv3_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_dconv3_relu = (wide_type*)q.enqueueMapBuffer(buffer_dconv3_relu, CL_TRUE, CL_MAP_READ, 0, (conv3_size) * sizeof(wide_type), NULL, NULL, &err));

    wide_type *ptr_conv4, *ptr_dconv4, *ptr_dconv4_relu;
    OCL_CHECK(err, cl::Buffer buffer_conv4(context, CL_MEM_READ_WRITE, (conv4_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_conv4 = (wide_type*)q.enqueueMapBuffer(buffer_conv4, CL_TRUE, CL_MAP_READ, 0, (conv4_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dconv4(context, CL_MEM_READ_WRITE, (conv4_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_dconv4 = (wide_type*)q.enqueueMapBuffer(buffer_dconv4, CL_TRUE, CL_MAP_READ, 0, (conv4_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dconv4_relu(context, CL_MEM_READ_WRITE, (conv4_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_dconv4_relu = (wide_type*)q.enqueueMapBuffer(buffer_dconv4_relu, CL_TRUE, CL_MAP_READ, 0, (conv4_size) * sizeof(wide_type), NULL, NULL, &err));

    wide_type *ptr_conv5, *ptr_dconv5;
    OCL_CHECK(err, cl::Buffer buffer_conv5(context, CL_MEM_READ_WRITE, (conv5_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_conv5 = (wide_type*)q.enqueueMapBuffer(buffer_conv5, CL_TRUE, CL_MAP_READ, 0, (conv5_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dconv5(context, CL_MEM_READ_WRITE, (conv5_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_dconv5 = (wide_type*)q.enqueueMapBuffer(buffer_dconv5, CL_TRUE, CL_MAP_READ, 0, (conv5_size) * sizeof(wide_type), NULL, NULL, &err));

    wide_type *ptr_fc1, *ptr_dfc1;
    unsigned output_size = CNN1D_FC_1_F_widened;
    unsigned fc1_size = CNN1D_FC_1_C * 2;
    OCL_CHECK(err, cl::Buffer buffer_fc1(context, CL_MEM_READ_WRITE, (output_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_fc1 = (wide_type*)q.enqueueMapBuffer(buffer_fc1, CL_TRUE, CL_MAP_WRITE, 0, (output_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dfc1(context, CL_MEM_READ_WRITE, (fc1_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_dfc1 = (wide_type*)q.enqueueMapBuffer(buffer_dfc1, CL_TRUE, CL_MAP_WRITE, 0, (fc1_size) * sizeof(wide_type), NULL, NULL, &err));

    wide_type *ptr_softmax;
    OCL_CHECK(err, cl::Buffer buffer_softmax(context, CL_MEM_READ_WRITE, (output_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_softmax = (wide_type*)q.enqueueMapBuffer(buffer_softmax, CL_TRUE, CL_MAP_WRITE, 0, (output_size) * sizeof(wide_type), NULL, NULL, &err));

    wide_type *ptr_dwfc1, *ptr_dwconv1, *ptr_dwconv2, *ptr_dwconv3, *ptr_dwconv4, *ptr_dwconv5;
    OCL_CHECK(err, cl::Buffer buffer_dwfc1(context, CL_MEM_READ_WRITE, (CNN1D_FC_1_C * CNN1D_FC_1_F_widened) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_dwfc1 = (wide_type*)q.enqueueMapBuffer(buffer_dwfc1, CL_TRUE, CL_MAP_WRITE, 0, (CNN1D_FC_1_C * CNN1D_FC_1_F_widened) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dwconv1(context, CL_MEM_READ_WRITE, (conv1_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_dwconv1 = (wide_type*)q.enqueueMapBuffer(buffer_dwconv1, CL_TRUE, CL_MAP_WRITE, 0, (conv1_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dwconv2(context, CL_MEM_READ_WRITE, (conv2_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_dwconv2 = (wide_type*)q.enqueueMapBuffer(buffer_dwconv2, CL_TRUE, CL_MAP_WRITE, 0, (conv2_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dwconv3(context, CL_MEM_READ_WRITE, (conv3_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_dwconv3 = (wide_type*)q.enqueueMapBuffer(buffer_dwconv3, CL_TRUE, CL_MAP_WRITE, 0, (conv3_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dwconv4(context, CL_MEM_READ_WRITE, (conv4_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_dwconv4 = (wide_type*)q.enqueueMapBuffer(buffer_dwconv4, CL_TRUE, CL_MAP_WRITE, 0, (conv4_size) * sizeof(wide_type), NULL, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_dwconv5(context, CL_MEM_READ_WRITE, (conv5_size) * sizeof(wide_type), NULL, &err));
    OCL_CHECK(err, ptr_dwconv5 = (wide_type*)q.enqueueMapBuffer(buffer_dwconv5, CL_TRUE, CL_MAP_WRITE, 0, (conv5_size) * sizeof(wide_type), NULL, NULL, &err));

    place_in_wide(x_ptr, ptr_input, CNN1D_CONV_1_C_im2col_padded, CNN1D_CONV_1_W, WIDE_LEN);

    label.resize(16, 0);
    place_in_vector(label.data(), y_ptr, 3); // Y

    out_file << "X padded:" << std::endl;
    print_output_wide(ptr_input, CNN1D_CONV_1_C_im2col_padded, CNN1D_CONV_1_W, WIDE_LEN, out_file);

    out_file << "Label:" << std::endl;
	print_vector(label.data(), 3, out_file);

    HLSNN_DataType *mp1_flatten = new HLSNN_DataType [CNN1D_CONV_1_F * CNN1D_CONV_1_K * CNN1D_CONV_2_W];
    HLSNN_DataType *conv2_flatten = new HLSNN_DataType [CNN1D_CONV_2_F * 2 * CNN1D_CONV_2_W];
    HLSNN_DataType *conv3_flatten = new HLSNN_DataType [CNN1D_CONV_3_F * 2 * CNN1D_CONV_3_W];
    HLSNN_DataType *conv4_flatten = new HLSNN_DataType [CNN1D_CONV_4_F * 2 * CNN1D_CONV_4_W];

    // [CNN_CONV_1_F, CNN_CONV_1_C_im2col_padded] X [CNN_CONV_1_C_im2col_padded, CNN_CONV_1_W]
    // [16, 32] X [32, 128]
    // [rows_in, cols_in] X [cols_in, cols_out]

    // Set up kernel arguments for FW and BW kernels (FPGA)
    int narg = 0;
    OCL_CHECK(err, err = krnl_conv1_fw.setArg(narg++, buffer_mp1));
    OCL_CHECK(err, err = krnl_conv1_fw.setArg(narg++, buffer_input));
    OCL_CHECK(err, err = krnl_conv1_fw.setArg(narg++, buffer_conv1w));
    OCL_CHECK(err, err = krnl_conv1_fw.setArg(narg++, buffer_conv1b));
    OCL_CHECK(err, err = krnl_conv1_fw.setArg(narg++, buffer_conv1_ix));
    OCL_CHECK(err, err = krnl_conv1_fw.setArg(narg++, 0));
    OCL_CHECK(err, err = krnl_conv1_fw.setArg(narg++, CNN1D_CONV_1_F));
    OCL_CHECK(err, err = krnl_conv1_fw.setArg(narg++, CNN1D_CONV_1_C_im2col_padded));
    OCL_CHECK(err, err = krnl_conv1_fw.setArg(narg++, CNN1D_CONV_1_W));
    OCL_CHECK(err, err = krnl_conv1_fw.setArg(narg++, CNN1D_CONV_1_W_conv));
    OCL_CHECK(err, err = krnl_conv1_fw.setArg(narg++, CNN1D_CONV_1_K));
    OCL_CHECK(err, err = krnl_conv1_fw.setArg(narg++, (CNN1D_CONV_1_MP_W / WIDE_LEN)));
    OCL_CHECK(err, err = krnl_conv1_fw.setArg(narg++, true));
    OCL_CHECK(err, err = krnl_conv1_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv1_fw.setArg(narg++, true));
    OCL_CHECK(err, err = krnl_conv1_fw.setArg(narg++, true));
    OCL_CHECK(err, err = krnl_conv1_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv1_fw.setArg(narg++, true));
    OCL_CHECK(err, err = krnl_conv1_fw.setArg(narg++, false));

    narg = 0;
    OCL_CHECK(err, err = krnl_conv2_fw.setArg(narg++, buffer_mp2));
    OCL_CHECK(err, err = krnl_conv2_fw.setArg(narg++, buffer_mp1));
    OCL_CHECK(err, err = krnl_conv2_fw.setArg(narg++, buffer_conv2w));
    OCL_CHECK(err, err = krnl_conv2_fw.setArg(narg++, buffer_conv2b));
    OCL_CHECK(err, err = krnl_conv2_fw.setArg(narg++, buffer_conv2_ix));
    OCL_CHECK(err, err = krnl_conv2_fw.setArg(narg++, 0));
    OCL_CHECK(err, err = krnl_conv2_fw.setArg(narg++, CNN1D_CONV_2_F));
    OCL_CHECK(err, err = krnl_conv2_fw.setArg(narg++, CNN1D_CONV_2_C_im2col));
    OCL_CHECK(err, err = krnl_conv2_fw.setArg(narg++, CNN1D_CONV_2_W));
    OCL_CHECK(err, err = krnl_conv2_fw.setArg(narg++, CNN1D_CONV_3_W_conv));
    OCL_CHECK(err, err = krnl_conv2_fw.setArg(narg++, CNN1D_CONV_3_K));
    OCL_CHECK(err, err = krnl_conv2_fw.setArg(narg++, (CNN1D_CONV_3_W_conv / WIDE_LEN)));
    OCL_CHECK(err, err = krnl_conv2_fw.setArg(narg++, true));
    OCL_CHECK(err, err = krnl_conv2_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv2_fw.setArg(narg++, true));
    OCL_CHECK(err, err = krnl_conv2_fw.setArg(narg++, true));
    OCL_CHECK(err, err = krnl_conv2_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv2_fw.setArg(narg++, true));
    OCL_CHECK(err, err = krnl_conv2_fw.setArg(narg++, false));

    narg = 0;
    OCL_CHECK(err, err = krnl_conv3_fw.setArg(narg++, buffer_conv3));
    OCL_CHECK(err, err = krnl_conv3_fw.setArg(narg++, buffer_mp2));
    OCL_CHECK(err, err = krnl_conv3_fw.setArg(narg++, buffer_conv3w));
    OCL_CHECK(err, err = krnl_conv3_fw.setArg(narg++, buffer_conv3b));
    OCL_CHECK(err, err = krnl_conv3_fw.setArg(narg++, buffer_conv3_ix));
    OCL_CHECK(err, err = krnl_conv3_fw.setArg(narg++, 0));
    OCL_CHECK(err, err = krnl_conv3_fw.setArg(narg++, CNN1D_CONV_3_F));
    OCL_CHECK(err, err = krnl_conv3_fw.setArg(narg++, CNN1D_CONV_3_C_im2col));
    OCL_CHECK(err, err = krnl_conv3_fw.setArg(narg++, CNN1D_CONV_3_W));
    OCL_CHECK(err, err = krnl_conv3_fw.setArg(narg++, (CNN1D_CONV_3_W_conv - 1)));
    OCL_CHECK(err, err = krnl_conv3_fw.setArg(narg++, CNN1D_CONV_4_K));
    OCL_CHECK(err, err = krnl_conv3_fw.setArg(narg++, (32 / WIDE_LEN)));
    OCL_CHECK(err, err = krnl_conv3_fw.setArg(narg++, true));
    OCL_CHECK(err, err = krnl_conv3_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv3_fw.setArg(narg++, true));
    OCL_CHECK(err, err = krnl_conv3_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv3_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv3_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv3_fw.setArg(narg++, false));

    narg = 0;
    OCL_CHECK(err, err = krnl_conv4_fw.setArg(narg++, buffer_conv4));
    OCL_CHECK(err, err = krnl_conv4_fw.setArg(narg++, buffer_conv3));
    OCL_CHECK(err, err = krnl_conv4_fw.setArg(narg++, buffer_conv4w));
    OCL_CHECK(err, err = krnl_conv4_fw.setArg(narg++, buffer_conv4b));
    OCL_CHECK(err, err = krnl_conv4_fw.setArg(narg++, buffer_conv4_ix));
    OCL_CHECK(err, err = krnl_conv4_fw.setArg(narg++, 0));
    OCL_CHECK(err, err = krnl_conv4_fw.setArg(narg++, CNN1D_CONV_4_F));
    OCL_CHECK(err, err = krnl_conv4_fw.setArg(narg++, CNN1D_CONV_4_C_im2col));
    OCL_CHECK(err, err = krnl_conv4_fw.setArg(narg++, CNN1D_CONV_4_W));
    OCL_CHECK(err, err = krnl_conv4_fw.setArg(narg++, (CNN1D_CONV_4_W_conv - 2)));
    OCL_CHECK(err, err = krnl_conv4_fw.setArg(narg++, CNN1D_CONV_5_K));
    OCL_CHECK(err, err = krnl_conv4_fw.setArg(narg++, (32 / WIDE_LEN)));
    OCL_CHECK(err, err = krnl_conv4_fw.setArg(narg++, true));
    OCL_CHECK(err, err = krnl_conv4_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv4_fw.setArg(narg++, true));
    OCL_CHECK(err, err = krnl_conv4_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv4_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv4_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv4_fw.setArg(narg++, false));


    narg = 0;
    OCL_CHECK(err, err = krnl_conv5_fw.setArg(narg++, buffer_conv5));
    OCL_CHECK(err, err = krnl_conv5_fw.setArg(narg++, buffer_conv4));
    OCL_CHECK(err, err = krnl_conv5_fw.setArg(narg++, buffer_conv5w));
    OCL_CHECK(err, err = krnl_conv5_fw.setArg(narg++, buffer_conv5b));
    OCL_CHECK(err, err = krnl_conv5_fw.setArg(narg++, buffer_conv5_ix));
    OCL_CHECK(err, err = krnl_conv5_fw.setArg(narg++, 0));
    OCL_CHECK(err, err = krnl_conv5_fw.setArg(narg++, CNN1D_CONV_5_F));
    OCL_CHECK(err, err = krnl_conv5_fw.setArg(narg++, CNN1D_CONV_5_C_im2col));
    OCL_CHECK(err, err = krnl_conv5_fw.setArg(narg++, CNN1D_CONV_5_W));
    OCL_CHECK(err, err = krnl_conv5_fw.setArg(narg++, (CNN1D_CONV_5_W_conv - 4)));
    OCL_CHECK(err, err = krnl_conv5_fw.setArg(narg++, CNN1D_CONV_5_K));
    OCL_CHECK(err, err = krnl_conv5_fw.setArg(narg++, (CNN1D_CONV_5_W / WIDE_LEN)));
    OCL_CHECK(err, err = krnl_conv5_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv5_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv5_fw.setArg(narg++, true));
    OCL_CHECK(err, err = krnl_conv5_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv5_fw.setArg(narg++, true));
    OCL_CHECK(err, err = krnl_conv5_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv5_fw.setArg(narg++, false));


    narg = 0;
    OCL_CHECK(err, err = krnl_fc1_fw.setArg(narg++, buffer_fc1));
    OCL_CHECK(err, err = krnl_fc1_fw.setArg(narg++, buffer_fc1w));
    OCL_CHECK(err, err = krnl_fc1_fw.setArg(narg++, buffer_conv5));
    OCL_CHECK(err, err = krnl_fc1_fw.setArg(narg++, buffer_fc1b));
    OCL_CHECK(err, err = krnl_fc1_fw.setArg(narg++, buffer_fc1_ix));
    OCL_CHECK(err, err = krnl_fc1_fw.setArg(narg++, 0));
    OCL_CHECK(err, err = krnl_fc1_fw.setArg(narg++, 1));
    OCL_CHECK(err, err = krnl_fc1_fw.setArg(narg++, CNN1D_FC_1_C));
    OCL_CHECK(err, err = krnl_fc1_fw.setArg(narg++, CNN1D_FC_1_F_widened));
    OCL_CHECK(err, err = krnl_fc1_fw.setArg(narg++, CNN1D_FC_1_F));
    OCL_CHECK(err, err = krnl_fc1_fw.setArg(narg++, 0));
    OCL_CHECK(err, err = krnl_fc1_fw.setArg(narg++, 0));
    OCL_CHECK(err, err = krnl_fc1_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_fc1_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_fc1_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_fc1_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_fc1_fw.setArg(narg++, true));
    OCL_CHECK(err, err = krnl_fc1_fw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_fc1_fw.setArg(narg++, true));

    narg = 0;
    OCL_CHECK(err, err = krnl_fc1_dx.setArg(narg++, buffer_softmax));
    OCL_CHECK(err, err = krnl_fc1_dx.setArg(narg++, buffer_fc1w));
    OCL_CHECK(err, err = krnl_fc1_dx.setArg(narg++, buffer_dfc1));
    OCL_CHECK(err, err = krnl_fc1_dx.setArg(narg++, buffer_fc1b));
    OCL_CHECK(err, err = krnl_fc1_dx.setArg(narg++, buffer_fc1_ix));
    OCL_CHECK(err, err = krnl_fc1_dx.setArg(narg++, 1));
    OCL_CHECK(err, err = krnl_fc1_dx.setArg(narg++, 1));
    OCL_CHECK(err, err = krnl_fc1_dx.setArg(narg++, CNN1D_FC_1_C));
    OCL_CHECK(err, err = krnl_fc1_dx.setArg(narg++, CNN1D_FC_1_F_widened));
    OCL_CHECK(err, err = krnl_fc1_dx.setArg(narg++, CNN1D_FC_1_C));
    OCL_CHECK(err, err = krnl_fc1_dx.setArg(narg++, 0));
    OCL_CHECK(err, err = krnl_fc1_dx.setArg(narg++, 0));
    OCL_CHECK(err, err = krnl_fc1_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_fc1_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_fc1_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_fc1_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_fc1_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_fc1_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_fc1_dx.setArg(narg++, false));

    narg = 0;
    OCL_CHECK(err, err = krnl_conv5_dx.setArg(narg++, buffer_dconv4_relu));
    OCL_CHECK(err, err = krnl_conv5_dx.setArg(narg++, buffer_dfc1));
    OCL_CHECK(err, err = krnl_conv5_dx.setArg(narg++, buffer_conv5w_tr));
    OCL_CHECK(err, err = krnl_conv5_dx.setArg(narg++, buffer_conv5b));
    OCL_CHECK(err, err = krnl_conv5_dx.setArg(narg++, buffer_conv5_ix));
    OCL_CHECK(err, err = krnl_conv5_dx.setArg(narg++, 0));
    OCL_CHECK(err, err = krnl_conv5_dx.setArg(narg++, CNN1D_CONV_5_C_im2col));
    OCL_CHECK(err, err = krnl_conv5_dx.setArg(narg++, CNN1D_CONV_5_F));
    OCL_CHECK(err, err = krnl_conv5_dx.setArg(narg++, 32));
    OCL_CHECK(err, err = krnl_conv5_dx.setArg(narg++, 32));
    OCL_CHECK(err, err = krnl_conv5_dx.setArg(narg++, 2));
    OCL_CHECK(err, err = krnl_conv5_dx.setArg(narg++, 32));
    OCL_CHECK(err, err = krnl_conv5_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv5_dx.setArg(narg++, true));
    OCL_CHECK(err, err = krnl_conv5_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv5_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv5_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv5_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv5_dx.setArg(narg++, true));

    narg = 0;
    OCL_CHECK(err, err = krnl_conv4_dx.setArg(narg++, buffer_dconv3_relu));
    OCL_CHECK(err, err = krnl_conv4_dx.setArg(narg++, buffer_dconv4_relu));
    OCL_CHECK(err, err = krnl_conv4_dx.setArg(narg++, buffer_conv4w_tr));
    OCL_CHECK(err, err = krnl_conv4_dx.setArg(narg++, buffer_conv4b));
    OCL_CHECK(err, err = krnl_conv4_dx.setArg(narg++, buffer_conv4_ix));
    OCL_CHECK(err, err = krnl_conv4_dx.setArg(narg++, 0));
    OCL_CHECK(err, err = krnl_conv4_dx.setArg(narg++, CNN1D_CONV_4_C_im2col));
    OCL_CHECK(err, err = krnl_conv4_dx.setArg(narg++, CNN1D_CONV_4_F));
    OCL_CHECK(err, err = krnl_conv4_dx.setArg(narg++, 32));
    OCL_CHECK(err, err = krnl_conv4_dx.setArg(narg++, 32));
    OCL_CHECK(err, err = krnl_conv4_dx.setArg(narg++, 2));
    OCL_CHECK(err, err = krnl_conv4_dx.setArg(narg++, 32));
    OCL_CHECK(err, err = krnl_conv4_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv4_dx.setArg(narg++, true));
    OCL_CHECK(err, err = krnl_conv4_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv4_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv4_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv4_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv4_dx.setArg(narg++, true));

    narg = 0;
    OCL_CHECK(err, err = krnl_conv3_dx.setArg(narg++, buffer_dmp2));
    OCL_CHECK(err, err = krnl_conv3_dx.setArg(narg++, buffer_dconv3_relu));
    OCL_CHECK(err, err = krnl_conv3_dx.setArg(narg++, buffer_conv3w_tr));
    OCL_CHECK(err, err = krnl_conv3_dx.setArg(narg++, buffer_conv3b));
    OCL_CHECK(err, err = krnl_conv3_dx.setArg(narg++, buffer_conv3_ix));
    OCL_CHECK(err, err = krnl_conv3_dx.setArg(narg++, 0));
    OCL_CHECK(err, err = krnl_conv3_dx.setArg(narg++, CNN1D_CONV_3_C_im2col));
    OCL_CHECK(err, err = krnl_conv3_dx.setArg(narg++, CNN1D_CONV_3_F));
    OCL_CHECK(err, err = krnl_conv3_dx.setArg(narg++, 32));
    OCL_CHECK(err, err = krnl_conv3_dx.setArg(narg++, 32));
    OCL_CHECK(err, err = krnl_conv3_dx.setArg(narg++, 2));
    OCL_CHECK(err, err = krnl_conv3_dx.setArg(narg++, 32));
    OCL_CHECK(err, err = krnl_conv3_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv3_dx.setArg(narg++, true));
    OCL_CHECK(err, err = krnl_conv3_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv3_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv3_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv3_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv3_dx.setArg(narg++, true));


    narg = 0;
    OCL_CHECK(err, err = krnl_conv2_dx.setArg(narg++, buffer_dmp1));
    OCL_CHECK(err, err = krnl_conv2_dx.setArg(narg++, buffer_dconv2));
    OCL_CHECK(err, err = krnl_conv2_dx.setArg(narg++, buffer_conv2w_tr));
    OCL_CHECK(err, err = krnl_conv2_dx.setArg(narg++, buffer_conv2b));
    OCL_CHECK(err, err = krnl_conv2_dx.setArg(narg++, buffer_conv2_ix));
    OCL_CHECK(err, err = krnl_conv2_dx.setArg(narg++, 0));
    OCL_CHECK(err, err = krnl_conv2_dx.setArg(narg++, CNN1D_CONV_2_C_im2col));
    OCL_CHECK(err, err = krnl_conv2_dx.setArg(narg++, CNN1D_CONV_2_F));
    OCL_CHECK(err, err = krnl_conv2_dx.setArg(narg++, 64));
    OCL_CHECK(err, err = krnl_conv2_dx.setArg(narg++, 64));
    OCL_CHECK(err, err = krnl_conv2_dx.setArg(narg++, 2));
    OCL_CHECK(err, err = krnl_conv2_dx.setArg(narg++, 64));
    OCL_CHECK(err, err = krnl_conv2_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv2_dx.setArg(narg++, true));
    OCL_CHECK(err, err = krnl_conv2_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv2_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv2_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv2_dx.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv2_dx.setArg(narg++, true));

    narg = 0;
    OCL_CHECK(err, err = krnl_fc1_dw.setArg(narg++, buffer_softmax));
    OCL_CHECK(err, err = krnl_fc1_dw.setArg(narg++, buffer_dwfc1));
    OCL_CHECK(err, err = krnl_fc1_dw.setArg(narg++, buffer_conv5));
    OCL_CHECK(err, err = krnl_fc1_dw.setArg(narg++, buffer_fc1b));
    OCL_CHECK(err, err = krnl_fc1_dw.setArg(narg++, buffer_fc1_ix));
    OCL_CHECK(err, err = krnl_fc1_dw.setArg(narg++, 2));
    OCL_CHECK(err, err = krnl_fc1_dw.setArg(narg++, 1));
    OCL_CHECK(err, err = krnl_fc1_dw.setArg(narg++, 512));
    OCL_CHECK(err, err = krnl_fc1_dw.setArg(narg++, 16));
    OCL_CHECK(err, err = krnl_fc1_dw.setArg(narg++, 0));
    OCL_CHECK(err, err = krnl_fc1_dw.setArg(narg++, 0));
    OCL_CHECK(err, err = krnl_fc1_dw.setArg(narg++, 0));
    OCL_CHECK(err, err = krnl_fc1_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_fc1_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_fc1_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_fc1_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_fc1_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_fc1_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_fc1_dw.setArg(narg++, false));

    narg = 0;
    OCL_CHECK(err, err = krnl_conv5_dw.setArg(narg++, buffer_conv4));
    OCL_CHECK(err, err = krnl_conv5_dw.setArg(narg++, buffer_dfc1));
    OCL_CHECK(err, err = krnl_conv5_dw.setArg(narg++, buffer_dwconv5));
    OCL_CHECK(err, err = krnl_conv5_dw.setArg(narg++, buffer_conv5b));
    OCL_CHECK(err, err = krnl_conv5_dw.setArg(narg++, buffer_conv5_ix));
    OCL_CHECK(err, err = krnl_conv5_dw.setArg(narg++, 1));
    OCL_CHECK(err, err = krnl_conv5_dw.setArg(narg++, CNN1D_CONV_5_C_im2col));
    OCL_CHECK(err, err = krnl_conv5_dw.setArg(narg++, CNN1D_CONV_5_F));
    OCL_CHECK(err, err = krnl_conv5_dw.setArg(narg++, CNN1D_CONV_5_W));
    OCL_CHECK(err, err = krnl_conv5_dw.setArg(narg++, CNN1D_CONV_5_W));
    OCL_CHECK(err, err = krnl_conv5_dw.setArg(narg++, CNN1D_CONV_5_K));
    OCL_CHECK(err, err = krnl_conv5_dw.setArg(narg++, CNN1D_CONV_5_F / WIDE_LEN));
    OCL_CHECK(err, err = krnl_conv5_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv5_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv5_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv5_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv5_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv5_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv5_dw.setArg(narg++, false));

    narg = 0;
    OCL_CHECK(err, err = krnl_conv4_dw.setArg(narg++, buffer_conv3));
    OCL_CHECK(err, err = krnl_conv4_dw.setArg(narg++, buffer_dconv4_relu));
    OCL_CHECK(err, err = krnl_conv4_dw.setArg(narg++, buffer_dwconv4));
    OCL_CHECK(err, err = krnl_conv4_dw.setArg(narg++, buffer_conv4b));
    OCL_CHECK(err, err = krnl_conv4_dw.setArg(narg++, buffer_conv4_ix));
    OCL_CHECK(err, err = krnl_conv4_dw.setArg(narg++, 1));
    OCL_CHECK(err, err = krnl_conv4_dw.setArg(narg++, CNN1D_CONV_4_C_im2col));
    OCL_CHECK(err, err = krnl_conv4_dw.setArg(narg++, CNN1D_CONV_4_F));
    OCL_CHECK(err, err = krnl_conv4_dw.setArg(narg++, CNN1D_CONV_4_W));
    OCL_CHECK(err, err = krnl_conv4_dw.setArg(narg++, CNN1D_CONV_4_W));
    OCL_CHECK(err, err = krnl_conv4_dw.setArg(narg++, CNN1D_CONV_4_K));
    OCL_CHECK(err, err = krnl_conv4_dw.setArg(narg++, CNN1D_CONV_4_F / WIDE_LEN));
    OCL_CHECK(err, err = krnl_conv4_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv4_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv4_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv4_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv4_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv4_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv4_dw.setArg(narg++, false));

    narg = 0;
    OCL_CHECK(err, err = krnl_conv3_dw.setArg(narg++, buffer_mp2));
    OCL_CHECK(err, err = krnl_conv3_dw.setArg(narg++, buffer_dconv3_relu));
    OCL_CHECK(err, err = krnl_conv3_dw.setArg(narg++, buffer_dwconv3));
    OCL_CHECK(err, err = krnl_conv3_dw.setArg(narg++, buffer_conv3b));
    OCL_CHECK(err, err = krnl_conv3_dw.setArg(narg++, buffer_conv3_ix));
    OCL_CHECK(err, err = krnl_conv3_dw.setArg(narg++, 1));
    OCL_CHECK(err, err = krnl_conv3_dw.setArg(narg++, CNN1D_CONV_3_C_im2col));
    OCL_CHECK(err, err = krnl_conv3_dw.setArg(narg++, CNN1D_CONV_3_F));
    OCL_CHECK(err, err = krnl_conv3_dw.setArg(narg++, CNN1D_CONV_3_W));
    OCL_CHECK(err, err = krnl_conv3_dw.setArg(narg++, CNN1D_CONV_3_W));
    OCL_CHECK(err, err = krnl_conv3_dw.setArg(narg++, CNN1D_CONV_3_K));
    OCL_CHECK(err, err = krnl_conv3_dw.setArg(narg++, CNN1D_CONV_3_F / WIDE_LEN));
    OCL_CHECK(err, err = krnl_conv3_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv3_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv3_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv3_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv3_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv3_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv3_dw.setArg(narg++, false));

    narg = 0;
    OCL_CHECK(err, err = krnl_conv2_dw.setArg(narg++, buffer_mp1));
    OCL_CHECK(err, err = krnl_conv2_dw.setArg(narg++, buffer_dconv2));
    OCL_CHECK(err, err = krnl_conv2_dw.setArg(narg++, buffer_dwconv2));
    OCL_CHECK(err, err = krnl_conv2_dw.setArg(narg++, buffer_conv2b));
    OCL_CHECK(err, err = krnl_conv2_dw.setArg(narg++, buffer_conv2_ix));
    OCL_CHECK(err, err = krnl_conv2_dw.setArg(narg++, 1));
    OCL_CHECK(err, err = krnl_conv2_dw.setArg(narg++, CNN1D_CONV_2_C_im2col));
    OCL_CHECK(err, err = krnl_conv2_dw.setArg(narg++, CNN1D_CONV_2_F));
    OCL_CHECK(err, err = krnl_conv2_dw.setArg(narg++, CNN1D_CONV_2_W));
    OCL_CHECK(err, err = krnl_conv2_dw.setArg(narg++, CNN1D_CONV_2_W));
    OCL_CHECK(err, err = krnl_conv2_dw.setArg(narg++, CNN1D_CONV_2_K));
    OCL_CHECK(err, err = krnl_conv2_dw.setArg(narg++, CNN1D_CONV_2_F / WIDE_LEN));
    OCL_CHECK(err, err = krnl_conv2_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv2_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv2_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv2_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv2_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv2_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv2_dw.setArg(narg++, false));

    narg = 0;
    OCL_CHECK(err, err = krnl_conv1_dw.setArg(narg++, buffer_input));
    OCL_CHECK(err, err = krnl_conv1_dw.setArg(narg++, buffer_dconv1));
    OCL_CHECK(err, err = krnl_conv1_dw.setArg(narg++, buffer_dwconv1));
    OCL_CHECK(err, err = krnl_conv1_dw.setArg(narg++, buffer_conv1b));
    OCL_CHECK(err, err = krnl_conv1_dw.setArg(narg++, buffer_conv1_ix));
    OCL_CHECK(err, err = krnl_conv1_dw.setArg(narg++, 1));
    OCL_CHECK(err, err = krnl_conv1_dw.setArg(narg++, CNN1D_CONV_1_C_im2col_padded));
    OCL_CHECK(err, err = krnl_conv1_dw.setArg(narg++, CNN1D_CONV_1_F));
    OCL_CHECK(err, err = krnl_conv1_dw.setArg(narg++, CNN1D_CONV_1_W));
    OCL_CHECK(err, err = krnl_conv1_dw.setArg(narg++, CNN1D_CONV_1_W));
    OCL_CHECK(err, err = krnl_conv1_dw.setArg(narg++, CNN1D_CONV_1_K));
    OCL_CHECK(err, err = krnl_conv1_dw.setArg(narg++, CNN1D_CONV_1_F / WIDE_LEN));
    OCL_CHECK(err, err = krnl_conv1_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv1_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv1_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv1_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv1_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv1_dw.setArg(narg++, false));
    OCL_CHECK(err, err = krnl_conv1_dw.setArg(narg++, false));

    // Set up kernel arguments for WU kernel (FPGA)
    unsigned lr_ = 0.001;
    narg = 0;
    OCL_CHECK(err, err = krnl_wu_fc1.setArg(narg++, buffer_fc1w));
    OCL_CHECK(err, err = krnl_wu_fc1.setArg(narg++, buffer_dwfc1));
    OCL_CHECK(err, err = krnl_wu_fc1.setArg(narg++, unsigned(CNN1D_FC_1_C * CNN1D_FC_1_F)));
    OCL_CHECK(err, err = krnl_wu_fc1.setArg(narg++, lr_));

    narg = 0;
    OCL_CHECK(err, err = krnl_wu_conv1.setArg(narg++, buffer_conv1w_tr));
    OCL_CHECK(err, err = krnl_wu_conv1.setArg(narg++, buffer_dwconv1));
    OCL_CHECK(err, err = krnl_wu_conv1.setArg(narg++, unsigned(CNN1D_CONV_1_F * CNN1D_CONV_1_C_im2col_padded)));
    OCL_CHECK(err, err = krnl_wu_conv1.setArg(narg++, lr_));

    narg = 0;
    OCL_CHECK(err, err = krnl_wu_conv2.setArg(narg++, buffer_conv2w_tr));
    OCL_CHECK(err, err = krnl_wu_conv2.setArg(narg++, buffer_dwconv2));
    OCL_CHECK(err, err = krnl_wu_conv2.setArg(narg++, unsigned(CNN1D_CONV_2_F * CNN1D_CONV_2_C_im2col)));
    OCL_CHECK(err, err = krnl_wu_conv2.setArg(narg++, lr_));

    narg = 0;
    OCL_CHECK(err, err = krnl_wu_conv3.setArg(narg++, buffer_conv3w_tr));
    OCL_CHECK(err, err = krnl_wu_conv3.setArg(narg++, buffer_dwconv3));
    OCL_CHECK(err, err = krnl_wu_conv3.setArg(narg++, unsigned(CNN1D_CONV_3_F * CNN1D_CONV_3_C_im2col)));
    OCL_CHECK(err, err = krnl_wu_conv3.setArg(narg++, lr_));

    narg = 0;
    OCL_CHECK(err, err = krnl_wu_conv4.setArg(narg++, buffer_conv4w_tr));
    OCL_CHECK(err, err = krnl_wu_conv4.setArg(narg++, buffer_dwconv4));
    OCL_CHECK(err, err = krnl_wu_conv4.setArg(narg++, unsigned(CNN1D_CONV_4_F * CNN1D_CONV_4_C_im2col)));
    OCL_CHECK(err, err = krnl_wu_conv4.setArg(narg++, lr_));

    narg = 0;
    OCL_CHECK(err, err = krnl_wu_conv5.setArg(narg++, buffer_conv5w_tr));
    OCL_CHECK(err, err = krnl_wu_conv5.setArg(narg++, buffer_dwconv5));
    OCL_CHECK(err, err = krnl_wu_conv5.setArg(narg++, unsigned(CNN1D_CONV_5_F * CNN1D_CONV_5_C_im2col)));
    OCL_CHECK(err, err = krnl_wu_conv5.setArg(narg++, lr_));

    
    // Setting up threads parameters and storing them for later execution
    float learning_rate = 0.01;

    unsigned fc1w_size_wide = CNN1D_FC_1_C * (CNN1D_FC_1_F_widened / WIDE_LEN);
    std::vector<pthread_t> sgd_threads;
    std::vector<SGD_thread_data_wide*> sgd_data_vector;
    unsigned threading_factor = 4;
    unsigned array_in_4 = int(fc1w_size_wide / threading_factor);
    for(unsigned i = 0; i < threading_factor; ++i){
        unsigned start = i * array_in_4;
        unsigned end = start + array_in_4;
        struct SGD_thread_data_wide *sgd_data = (struct SGD_thread_data_wide *)malloc(sizeof(struct SGD_thread_data_wide));
        sgd_data->w_orig = ptr_fc1w;
        sgd_data->w = ptr_dwfc1;
        sgd_data->LR = learning_rate;
        sgd_data->length = fc1w_size_wide;
        sgd_data->start = start;
        sgd_data->end = end;
        sgd_data->wide_length = WIDE_LEN;
        pthread_t sgd_thrd;
        sgd_data_vector.push_back(sgd_data);
        sgd_threads.push_back(sgd_thrd);
    }

    unsigned conv5_size_wide = CNN1D_CONV_5_F * (CNN1D_CONV_5_C_im2col / WIDE_LEN);
    std::vector<pthread_t> sgd_threads_conv5;
    std::vector<SGD_thread_data_wide*> sgd_data_vector_conv5;
    unsigned array_in_4_conv5 = int(conv5_size_wide / threading_factor);
    for(unsigned i = 0; i < threading_factor; ++i){
        unsigned start = i * array_in_4_conv5;
        unsigned end = start + array_in_4_conv5;
        struct SGD_thread_data_wide *sgd_data = (struct SGD_thread_data_wide *)malloc(sizeof(struct SGD_thread_data_wide));
        sgd_data->w_orig = ptr_conv5w_tr;
        sgd_data->w = ptr_dwconv5;
        sgd_data->LR = learning_rate;
        sgd_data->length = conv5_size_wide;
        sgd_data->start = start;
        sgd_data->end = end;
        sgd_data->wide_length = WIDE_LEN;
        pthread_t sgd_thrd;
        sgd_data_vector_conv5.push_back(sgd_data);
        sgd_threads_conv5.push_back(sgd_thrd);
    }

    unsigned conv4_size_wide = CNN1D_CONV_4_F * (CNN1D_CONV_4_C_im2col / WIDE_LEN);
    std::vector<pthread_t> sgd_threads_conv4;
    std::vector<SGD_thread_data_wide*> sgd_data_vector_conv4;
    unsigned array_in_4_conv4 = int(conv4_size_wide / threading_factor);
    for(unsigned i = 0; i < threading_factor; ++i){
        unsigned start = i * array_in_4_conv4;
        unsigned end = start + array_in_4_conv4;
        struct SGD_thread_data_wide *sgd_data = (struct SGD_thread_data_wide *)malloc(sizeof(struct SGD_thread_data_wide));
        sgd_data->w_orig = ptr_conv4w_tr;
        sgd_data->w = ptr_dwconv4;
        sgd_data->LR = learning_rate;
        sgd_data->length = conv4_size_wide;
        sgd_data->start = start;
        sgd_data->end = end;
        sgd_data->wide_length = WIDE_LEN;
        pthread_t sgd_thrd;
        sgd_data_vector_conv4.push_back(sgd_data);
        sgd_threads_conv4.push_back(sgd_thrd);
    }

    unsigned conv3_size_wide = CNN1D_CONV_3_F * (CNN1D_CONV_3_C_im2col / WIDE_LEN);
    std::vector<pthread_t> sgd_threads_conv3;
    std::vector<SGD_thread_data_wide*> sgd_data_vector_conv3;
    unsigned array_in_4_conv3 = int(conv3_size_wide / threading_factor);
    for(unsigned i = 0; i < threading_factor; ++i){
        unsigned start = i * array_in_4_conv3;
        unsigned end = start + array_in_4_conv3;
        struct SGD_thread_data_wide *sgd_data = (struct SGD_thread_data_wide *)malloc(sizeof(struct SGD_thread_data_wide));
        sgd_data->w_orig = ptr_conv3w_tr;
        sgd_data->w = ptr_dwconv3;
        sgd_data->LR = learning_rate;
        sgd_data->length = conv3_size_wide;
        sgd_data->start = start;
        sgd_data->end = end;
        sgd_data->wide_length = WIDE_LEN;
        pthread_t sgd_thrd;
        sgd_data_vector_conv3.push_back(sgd_data);
        sgd_threads_conv3.push_back(sgd_thrd);
    }

    unsigned conv2_size_wide = CNN1D_CONV_2_F * (CNN1D_CONV_2_C_im2col / WIDE_LEN);
    std::vector<pthread_t> sgd_threads_conv2;
    std::vector<SGD_thread_data_wide*> sgd_data_vector_conv2;
    unsigned array_in_4_conv2 = int(conv2_size_wide / threading_factor);
    for(unsigned i = 0; i < threading_factor; ++i){
        unsigned start = i * array_in_4_conv2;
        unsigned end = start + array_in_4_conv2;
        struct SGD_thread_data_wide *sgd_data = (struct SGD_thread_data_wide *)malloc(sizeof(struct SGD_thread_data_wide));
        sgd_data->w_orig = ptr_conv2w_tr;
        sgd_data->w = ptr_dwconv2;
        sgd_data->LR = learning_rate;
        sgd_data->length = conv2_size_wide;
        sgd_data->start = start;
        sgd_data->end = end;
        sgd_data->wide_length = WIDE_LEN;
        pthread_t sgd_thrd;
        sgd_data_vector_conv2.push_back(sgd_data);
        sgd_threads_conv2.push_back(sgd_thrd);
    }

    unsigned conv1_size_wide = CNN1D_CONV_1_F * (CNN1D_CONV_1_C_im2col_padded / WIDE_LEN);
    std::vector<pthread_t> sgd_threads_conv1;
    std::vector<SGD_thread_data_wide*> sgd_data_vector_conv1;
    unsigned array_in_4_conv1 = int(conv1_size_wide / threading_factor);
    for(unsigned i = 0; i < threading_factor; ++i){
        unsigned start = i * array_in_4_conv1;
        unsigned end = start + array_in_4_conv1;
        struct SGD_thread_data_wide *sgd_data = (struct SGD_thread_data_wide *)malloc(sizeof(struct SGD_thread_data_wide));
        sgd_data->w_orig = ptr_conv1w_tr;
        sgd_data->w = ptr_dwconv1;
        sgd_data->LR = learning_rate;
        sgd_data->length = conv1_size_wide;
        sgd_data->start = start;
        sgd_data->end = end;
        sgd_data->wide_length = WIDE_LEN;
        pthread_t sgd_thrd;
        sgd_data_vector_conv1.push_back(sgd_data);
        sgd_threads_conv1.push_back(sgd_thrd);
    }

    std::vector<pthread_t> dconv1_threads;
    std::vector<mp_relu_bw_thread_data*> dconv1_data;
    unsigned dconv1_size = 32;
    unsigned array_in_4_dconv1 = int(dconv1_size / threading_factor);
    for(unsigned i = 0; i < threading_factor; ++i){
        unsigned start = i * array_in_4_dconv1;
        unsigned end = start + array_in_4_dconv1;
        struct mp_relu_bw_thread_data *dconv_data = (struct mp_relu_bw_thread_data *)malloc(sizeof(struct mp_relu_bw_thread_data));
        dconv_data->gradient_in = ptr_dmp1;
        dconv_data->gradient_out = ptr_dconv1;
        dconv_data->mp_out = mp1_flatten;
        dconv_data->indices = ptr_conv1_ix;
        dconv_data->M = 32;
        dconv_data->N = 128;
        dconv_data->K = 2;
        dconv_data->C = 64;
        dconv_data->wide_length = WIDE_LEN;
        dconv_data->start = start;
        dconv_data->end = end;
        pthread_t dconv_thr;
        dconv1_data.push_back(dconv_data);
        dconv1_threads.push_back(dconv_thr);
    }

    std::vector<pthread_t> dconv2_threads;
    std::vector<mp_relu_bw_thread_data*> dconv2_data;
    unsigned dconv2_size = 32;
    unsigned array_in_4_dconv2 = int(dconv2_size / threading_factor);
    for(unsigned i = 0; i < threading_factor; ++i){
        unsigned start = i * array_in_4_dconv2;
        unsigned end = start + array_in_4_dconv2;
        struct mp_relu_bw_thread_data *dconv_data = (struct mp_relu_bw_thread_data *)malloc(sizeof(struct mp_relu_bw_thread_data));
        dconv_data->gradient_in = ptr_dmp2;
        dconv_data->gradient_out = ptr_dconv2;
        dconv_data->mp_out = conv2_flatten;
        dconv_data->indices = ptr_conv2_ix;
        dconv_data->M = 32;
        dconv_data->N = 64;
        dconv_data->K = 3;
        dconv_data->C = 32;
        dconv_data->wide_length = WIDE_LEN;
        dconv_data->start = start;
        dconv_data->end = end;
        pthread_t dconv_thr;
        dconv2_data.push_back(dconv_data);
        dconv2_threads.push_back(dconv_thr);
    }

    std::vector<pthread_t> drelu3_threads;
    std::vector<relu_bw_thread_data*> drelu3_data;
    unsigned drelu3_size = CNN1D_CONV_4_F;
    unsigned array_in_4_drelu3 = int(drelu3_size / threading_factor);
    for(unsigned i = 0; i < threading_factor; ++i){
        unsigned start = i * array_in_4_drelu3;
        unsigned end = start + array_in_4_drelu3;
        struct relu_bw_thread_data *drelu_data = (struct relu_bw_thread_data *)malloc(sizeof(struct relu_bw_thread_data));
        drelu_data->gradient_out = ptr_dconv3_relu;
        drelu_data->activation_in = conv3_flatten;
        drelu_data->M = CNN1D_CONV_4_F;
        drelu_data->N = CNN1D_CONV_4_W;
        drelu_data->K = CNN1D_CONV_4_K;
        drelu_data->C = 16;
        drelu_data->wide_length = WIDE_LEN;
        drelu_data->start = start;
        drelu_data->end = end;
        pthread_t drelu_thr;
        drelu3_data.push_back(drelu_data);
        drelu3_threads.push_back(drelu_thr);
    }

    std::vector<pthread_t> drelu4_threads;
    std::vector<relu_bw_thread_data*> drelu4_data;
    unsigned drelu4_size = CNN1D_CONV_5_F;
    unsigned array_in_4_drelu4 = int(drelu4_size / threading_factor);
    for(unsigned i = 0; i < threading_factor; ++i){
        unsigned start = i * array_in_4_drelu4;
        unsigned end = start + array_in_4_drelu4;
        struct relu_bw_thread_data *drelu_data = (struct relu_bw_thread_data *)malloc(sizeof(struct relu_bw_thread_data));
        drelu_data->gradient_out = ptr_dconv4_relu;
        drelu_data->activation_in = conv4_flatten;
        drelu_data->M = CNN1D_CONV_5_F;
        drelu_data->N = CNN1D_CONV_5_W;
        drelu_data->K = CNN1D_CONV_5_K;
        drelu_data->C = 16;
        drelu_data->wide_length = WIDE_LEN;
        drelu_data->start = start;
        drelu_data->end = end;
        pthread_t drelu_thr;
        drelu4_data.push_back(drelu_data);
        drelu4_threads.push_back(drelu_thr);
    }

    std::vector<pthread_t> drelu5_threads;
    std::vector<relu_bw_simple_thread_data*> drelu5_data;
    unsigned drelu5_size = CNN1D_CONV_5_F;
    unsigned array_in_4_drelu5 = int(drelu5_size / threading_factor);
    for(unsigned i = 0; i < threading_factor; ++i){
        unsigned start = i * array_in_4_drelu5;
        unsigned end = start + array_in_4_drelu5;
        struct relu_bw_simple_thread_data *drelu_data = (struct relu_bw_simple_thread_data *)malloc(sizeof(struct relu_bw_simple_thread_data));
        drelu_data->gradient = ptr_dfc1;
        drelu_data->activation = ptr_conv5;
        drelu_data->M = CNN1D_CONV_5_F;
        drelu_data->N = CNN1D_CONV_5_W;
        drelu_data->wide_length = WIDE_LEN;
        drelu_data->start = start;
        drelu_data->end = end;
        pthread_t drelu_thr;
        drelu5_data.push_back(drelu_data);
        drelu5_threads.push_back(drelu_thr);
    }

    std::vector<pthread_t> dbconv1_threads;
    std::vector<dB_SGD_thread_data*> dbconv1_data;
    unsigned dbconv1_size = CNN1D_CONV_1_F;
    unsigned array_in_4_dbconv1 = int(dbconv1_size / threading_factor);
    for(unsigned i = 0; i < threading_factor; ++i){
        unsigned start = i * array_in_4_dbconv1;
        unsigned end = start + array_in_4_dbconv1;
        struct dB_SGD_thread_data *dbconv_data = (struct dB_SGD_thread_data *)malloc(sizeof(struct dB_SGD_thread_data));
        dbconv_data->ofm = ptr_dconv1;
        dbconv_data->b = ptr_conv1b;
        dbconv_data->C = CNN1D_CONV_1_F;
        dbconv_data->C_orig = CNN1D_CONV_1_F;
        dbconv_data->W = CNN1D_CONV_1_W;
        dbconv_data->learning_rate = learning_rate;
        dbconv_data->wide_length = WIDE_LEN;
        dbconv_data->start = start;
        dbconv_data->end = end;
        pthread_t dbconv_thr;
        dbconv1_data.push_back(dbconv_data);
        dbconv1_threads.push_back(dbconv_thr);
    }


    //##############################################################################
    std::cout << "Start process" << std::endl;
    auto all_begin = std::chrono::high_resolution_clock::now();
    auto fw_begin = std::chrono::high_resolution_clock::now();

    // ################################## FORWARD PASS ##################################
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({
                            buffer_mp1, buffer_mp2, buffer_conv3, buffer_conv4, buffer_conv5, buffer_fc1, buffer_input,
                            buffer_conv1w, buffer_conv1b, buffer_conv1_ix,
                            buffer_conv2w, buffer_conv2b, buffer_conv2_ix,
                            buffer_conv3w, buffer_conv3b, buffer_conv3_ix,
                            buffer_conv4w, buffer_conv4b, buffer_conv4_ix,
                            buffer_conv5w, buffer_conv5b, buffer_conv5_ix,
                            buffer_fc1w, buffer_fc1b, buffer_fc1_ix,
                            buffer_dfc1}, 0));
    auto conv1_fw_begin = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueTask(krnl_conv1_fw));
    OCL_CHECK(err, q.finish());
    auto conv1_fw_end = std::chrono::high_resolution_clock::now();
    
    auto conv2_fw_begin = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueTask(krnl_conv2_fw));
    OCL_CHECK(err, q.finish());
    auto conv2_fw_end = std::chrono::high_resolution_clock::now();
    
    auto conv3_fw_begin = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueTask(krnl_conv3_fw));
    OCL_CHECK(err, q.finish());
    auto conv3_fw_end = std::chrono::high_resolution_clock::now();

    auto conv4_fw_begin = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueTask(krnl_conv4_fw));
    OCL_CHECK(err, q.finish());
    auto conv4_fw_end = std::chrono::high_resolution_clock::now();

    auto conv5_fw_begin = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueTask(krnl_conv5_fw));
    OCL_CHECK(err, q.finish());
    auto conv5_fw_end = std::chrono::high_resolution_clock::now();

    auto fc1_fw_begin = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueTask(krnl_fc1_fw));
    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_mp1, buffer_mp2, buffer_conv3, buffer_conv4, buffer_conv5, buffer_fc1}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, q.finish());
    auto fc1_fw_end = std::chrono::high_resolution_clock::now();
    
    auto fw_end = std::chrono::high_resolution_clock::now();
    //################################## SOFTMAX ##################################

    auto softmax_begin = std::chrono::high_resolution_clock::now();
    compute_softmax(ptr_fc1, label, ptr_softmax, 1, 3, WIDE_LEN);
    auto softmax_end = std::chrono::high_resolution_clock::now();

    auto conv1_reverse_im2col_begin = std::chrono::high_resolution_clock::now();
    reverse_im2col_and_flatten(ptr_mp1, mp1_flatten, CNN1D_CONV_1_F, CNN1D_CONV_1_W, CNN1D_CONV_2_W, CNN1D_CONV_1_K, WIDE_LEN);
    auto conv1_reverse_im2col_end = std::chrono::high_resolution_clock::now();

    auto conv2_reverse_im2col_begin = std::chrono::high_resolution_clock::now();
    reverse_im2col_and_flatten(ptr_mp2, conv2_flatten, CNN1D_CONV_2_F, CNN1D_CONV_2_W, CNN1D_CONV_2_W_conv - 2, 2, WIDE_LEN);
    auto conv2_reverse_im2col_end = std::chrono::high_resolution_clock::now();

    auto conv3_reverse_im2col_begin = std::chrono::high_resolution_clock::now();
    reverse_im2col_and_flatten2(ptr_conv3, conv3_flatten, CNN1D_CONV_3_F, CNN1D_CONV_3_W, CNN1D_CONV_3_W_conv - 2, 2, WIDE_LEN);
    auto conv3_reverse_im2col_end = std::chrono::high_resolution_clock::now();

    auto conv4_reverse_im2col_begin = std::chrono::high_resolution_clock::now();
    reverse_im2col_and_flatten2(ptr_conv4, conv4_flatten, CNN1D_CONV_4_F, CNN1D_CONV_4_W, CNN1D_CONV_4_W_conv - 2, 2, WIDE_LEN);
    auto conv4_reverse_im2col_end = std::chrono::high_resolution_clock::now();

    // ################################## BACKWARD PASS ##################################
    auto bw_begin = std::chrono::high_resolution_clock::now();

    auto dx_begin = std::chrono::high_resolution_clock::now();
    //################################## dX FC1 ##################################
    auto dx_fc1_begin = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_softmax}, 0 ));
    OCL_CHECK(err, err = q.enqueueTask(krnl_fc1_dx));
    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_dfc1}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, q.finish());
    auto dx_fc1_end = std::chrono::high_resolution_clock::now();

    //################################## dX CONV5 (ReLU) ##################################
    auto dx_conv5_relu_begin = std::chrono::high_resolution_clock::now();
    //Uncomment this for 1 thread execution, and comment the thread calls.
    //relu_bw(ptr_dfc1, ptr_conv5, CNN2_CONV_5_F, CNN2_CONV_5_W, WIDE_LEN);

    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_create(&(drelu5_threads[i]), NULL, worker_relu_bw, (void *)drelu5_data[i]);
    }
            
    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_join(drelu5_threads[i], NULL);
    }
    auto dx_conv5_relu_end = std::chrono::high_resolution_clock::now();

    //################################## dX CONV5 (Conv) ##################################
    auto dx_conv5_begin = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_dfc1, buffer_dconv4_relu}, 0 ));
    OCL_CHECK(err, err = q.enqueueTask(krnl_conv5_dx));
    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_dconv4_relu}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, q.finish());
    auto dx_conv5_end = std::chrono::high_resolution_clock::now();

    //################################## dX CONV4 (ReLU) ##################################
    auto dx_conv4_relu_begin = std::chrono::high_resolution_clock::now();
    //Uncomment this for 1 thread execution, and comment the thread calls.
    //relu_bw_im2col(ptr_dconv4_relu, conv4_flatten, CNN2_CONV_5_F, CNN2_CONV_5_W, CNN2_CONV_5_K, 16, WIDE_LEN);

    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_create(&(drelu4_threads[i]), NULL, worker_relu_bw_im2col, (void *)drelu4_data[i]);
    }
            
    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_join(drelu4_threads[i], NULL);
    }
    auto dx_conv4_relu_end = std::chrono::high_resolution_clock::now();

    //################################## dX CONV4 (Conv) ##################################
    auto dx_conv4_begin = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_dconv4_relu, buffer_dconv3_relu}, 0 ));
    OCL_CHECK(err, err = q.enqueueTask(krnl_conv4_dx));
    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_dconv3_relu}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, q.finish());
    auto dx_conv4_end = std::chrono::high_resolution_clock::now();

    //################################## dX CONV3 (ReLU) ##################################
    auto dx_conv3_relu_begin = std::chrono::high_resolution_clock::now();
    //Uncomment this for 1 thread execution, and comment the thread calls.
    //relu_bw_im2col(ptr_dconv3_relu, conv3_flatten, CNN2_CONV_4_F, CNN2_CONV_4_W, CNN2_CONV_4_K, 16, WIDE_LEN);

    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_create(&(drelu3_threads[i]), NULL, worker_relu_bw_im2col, (void *)drelu3_data[i]);
    }
            
    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_join(drelu3_threads[i], NULL);
    }
    auto dx_conv3_relu_end = std::chrono::high_resolution_clock::now();

    //################################## dX CONV3 (Conv) ##################################
    auto dx_conv3_begin = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_dconv3_relu, buffer_dmp2}, 0 ));
    OCL_CHECK(err, err = q.enqueueTask(krnl_conv3_dx));
    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_dmp2}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, q.finish());
    auto dx_conv3_end = std::chrono::high_resolution_clock::now();

    //################################## dX CONV2 (ReLU + MP) ##################################
    auto dx_conv2_mp_relu_begin = std::chrono::high_resolution_clock::now();
    //Uncomment this for 1 thread execution, and comment the thread calls.
    //maxpool_relu_bw_im2col(ptr_dmp2, ptr_dconv2, conv2_flatten, ptr_conv2_ix, 32, 64, 3, 32, WIDE_LEN);

    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_create(&(dconv2_threads[i]), NULL, worker_maxpool_relu_bw_im2col, (void *)dconv2_data[i]);
    }
            
    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_join(dconv2_threads[i], NULL);
    }
    auto dx_conv2_mp_relu_end = std::chrono::high_resolution_clock::now();

    //################################## dX CONV2 (Conv) ##################################
    auto dx_conv2_begin = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_dconv2, buffer_dmp1}, 0 ));
    OCL_CHECK(err, err = q.enqueueTask(krnl_conv2_dx));
    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_dmp1}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, q.finish());
    auto dx_conv2_end = std::chrono::high_resolution_clock::now();

    auto dx_conv1_begin = std::chrono::high_resolution_clock::now();
    //Uncomment this for 1 thread execution, and comment the thread calls.
    //maxpool_relu_bw_im2col(ptr_dmp1, ptr_dconv1, mp1_flatten, ptr_conv1_ix, 32, 128, 2, 64, WIDE_LEN);
    
    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_create(&(dconv1_threads[i]), NULL, worker_maxpool_relu_bw_im2col, (void *)dconv1_data[i]);
    }
            
    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_join(dconv1_threads[i], NULL);
    }
    auto dx_conv1_end = std::chrono::high_resolution_clock::now();
    auto dx_end = std::chrono::high_resolution_clock::now();


    //=======DW================================================================================================================
    out_file << std::endl << "################################## dW FC1 ##################################" << std::endl;

    auto dw_begin = std::chrono::high_resolution_clock::now();
    auto dw_fc1_begin = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_softmax, buffer_dwfc1, buffer_conv5}, 0 ));
    OCL_CHECK(err, err = q.enqueueTask(krnl_fc1_dw));
    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_dwfc1}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, q.finish());
    auto dw_fc1_end = std::chrono::high_resolution_clock::now();

    auto dw_conv5_begin = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_conv4, buffer_dfc1, buffer_dwconv5}, 0 ));
    OCL_CHECK(err, err = q.enqueueTask(krnl_conv5_dw));
    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_dwconv5}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, q.finish());
    auto dw_conv5_end = std::chrono::high_resolution_clock::now();

    auto dw_conv4_begin = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_conv3, buffer_dconv4_relu, buffer_dwconv4}, 0 ));
    OCL_CHECK(err, err = q.enqueueTask(krnl_conv4_dw));
    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_dwconv4}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, q.finish());
    auto dw_conv4_end = std::chrono::high_resolution_clock::now();

    auto dw_conv3_begin = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_mp2, buffer_dconv3_relu, buffer_dwconv3}, 0 ));
    OCL_CHECK(err, err = q.enqueueTask(krnl_conv3_dw));
    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_dwconv3}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, q.finish());
    auto dw_conv3_end = std::chrono::high_resolution_clock::now();

    auto dw_conv2_begin = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_mp1, buffer_dconv2, buffer_dwconv2}, 0 ));
    OCL_CHECK(err, err = q.enqueueTask(krnl_conv2_dw));
    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_dwconv2}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, q.finish());
    auto dw_conv2_end = std::chrono::high_resolution_clock::now();

    auto dw_conv1_begin = std::chrono::high_resolution_clock::now();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input, buffer_dconv1, buffer_dwconv1}, 0 ));
    OCL_CHECK(err, err = q.enqueueTask(krnl_conv1_dw));
    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_dwconv1}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, q.finish());
    auto dw_conv1_end = std::chrono::high_resolution_clock::now();
    auto dw_end = std::chrono::high_resolution_clock::now();

    //#########################WU###############################################
    auto wu_begin = std::chrono::high_resolution_clock::now();
    auto wufc1_begin = std::chrono::high_resolution_clock::now();
    //Uncomment this for 1 thread execution, and comment the thread calls.
    //weight_sgd(ptr_fc1w, ptr_dwfc1, CNN2_FC_1_C, CNN2_FC_1_F_widened, learning_rate, WIDE_LEN);
    
    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_create(&(sgd_threads[i]), NULL, worker_SGD_wide, (void *)sgd_data_vector[i]);
    }
            
    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_join(sgd_threads[i], NULL);
    }
    
    auto wufc1_end = std::chrono::high_resolution_clock::now();

    auto wuconv5_begin = std::chrono::high_resolution_clock::now();
    //Uncomment this for 1 thread execution, and comment the thread calls.
    //weight_sgd(ptr_conv5w_tr, ptr_dwconv5, CNN2_CONV_5_C_im2col, CNN2_CONV_5_F, learning_rate, WIDE_LEN);
    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_create(&(sgd_threads_conv5[i]), NULL, worker_SGD_wide, (void *)sgd_data_vector_conv5[i]);
    }
            
    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_join(sgd_threads_conv5[i], NULL);
    }
    auto wuconv5_end = std::chrono::high_resolution_clock::now();

    auto conv5_tr_begin = std::chrono::high_resolution_clock::now();
    transpose_matrix_widened(ptr_conv5w_tr, ptr_conv5w, CNN1D_CONV_5_C_im2col, CNN1D_CONV_5_F, WIDE_LEN);
    auto conv5_tr_end = std::chrono::high_resolution_clock::now();

    auto wuconv4_begin = std::chrono::high_resolution_clock::now();
    //Uncomment this for 1 thread execution, and comment the thread calls.
    //weight_sgd(ptr_conv4w_tr, ptr_dwconv4, CNN2_CONV_4_C_im2col, CNN2_CONV_4_F, learning_rate, WIDE_LEN);
    
    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_create(&(sgd_threads_conv4[i]), NULL, worker_SGD_wide, (void *)sgd_data_vector_conv4[i]);
    }
            
    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_join(sgd_threads_conv4[i], NULL);
    }
    auto wuconv4_end = std::chrono::high_resolution_clock::now();

    auto conv4_tr_begin = std::chrono::high_resolution_clock::now();
    transpose_matrix_widened(ptr_conv4w_tr, ptr_conv4w, CNN1D_CONV_4_C_im2col, CNN1D_CONV_4_F, WIDE_LEN);
    auto conv4_tr_end = std::chrono::high_resolution_clock::now();

    auto wuconv3_begin = std::chrono::high_resolution_clock::now();
    //Uncomment this for 1 thread execution, and comment the thread calls.
    //weight_sgd(ptr_conv3w_tr, ptr_dwconv3, CNN2_CONV_3_C_im2col, CNN2_CONV_3_F, learning_rate, WIDE_LEN);

    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_create(&(sgd_threads_conv3[i]), NULL, worker_SGD_wide, (void *)sgd_data_vector_conv3[i]);
    }
            
    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_join(sgd_threads_conv3[i], NULL);
    }
    auto wuconv3_end = std::chrono::high_resolution_clock::now();

    auto conv3_tr_begin = std::chrono::high_resolution_clock::now();
    transpose_matrix_widened(ptr_conv3w_tr, ptr_conv3w, CNN1D_CONV_3_C_im2col, CNN1D_CONV_3_F, WIDE_LEN);
    auto conv3_tr_end = std::chrono::high_resolution_clock::now();

    auto wuconv2_begin = std::chrono::high_resolution_clock::now();
    //Uncomment this for 1 thread execution, and comment the thread calls.
    //weight_sgd(ptr_conv2w_tr, ptr_dwconv2, CNN2_CONV_2_C_im2col, CNN2_CONV_2_F, learning_rate, WIDE_LEN);
    
    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_create(&(sgd_threads_conv2[i]), NULL, worker_SGD_wide, (void *)sgd_data_vector_conv2[i]);
    }
            
    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_join(sgd_threads_conv2[i], NULL);
    }
    auto wuconv2_end = std::chrono::high_resolution_clock::now();

    auto conv2_tr_begin = std::chrono::high_resolution_clock::now();
    transpose_matrix_widened(ptr_conv2w_tr, ptr_conv2w, CNN1D_CONV_2_C_im2col, CNN1D_CONV_2_F, WIDE_LEN);
    auto conv2_tr_end = std::chrono::high_resolution_clock::now();

    auto wuconv1_begin = std::chrono::high_resolution_clock::now();
    //Uncomment this for 1 thread execution, and comment the thread calls.
    //weight_sgd(ptr_conv1w_tr, ptr_dwconv1, CNN2_CONV_1_C_im2col, CNN2_CONV_1_F, learning_rate, WIDE_LEN);

    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_create(&(sgd_threads_conv1[i]), NULL, worker_SGD_wide, (void *)sgd_data_vector_conv1[i]);
    }
            
    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_join(sgd_threads_conv1[i], NULL);
    }
    auto wuconv1_end = std::chrono::high_resolution_clock::now();
    
    auto conv1_tr_begin = std::chrono::high_resolution_clock::now();
    transpose_matrix_widened(ptr_conv1w_tr, ptr_conv1w, CNN1D_CONV_1_C_im2col_padded, CNN1D_CONV_1_F, WIDE_LEN);
    auto conv1_tr_end = std::chrono::high_resolution_clock::now();
    
    auto wu_end = std::chrono::high_resolution_clock::now();

    //################################## BIAS UPDATES ##################################
    auto bu_begin = std::chrono::high_resolution_clock::now();
    
    auto bufc1_begin = std::chrono::high_resolution_clock::now();
    bias_sgd_FC(ptr_fc1b, ptr_softmax, 1, CNN1D_FC_1_F_widened, CNN1D_FC_1_F, learning_rate, WIDE_LEN);
    auto bufc1_end = std::chrono::high_resolution_clock::now();

    auto buconv5_begin = std::chrono::high_resolution_clock::now();
    do_dB_conv(ptr_dfc1, ptr_conv5b, CNN1D_CONV_5_F, CNN1D_CONV_5_F, CNN1D_CONV_5_W, learning_rate, WIDE_LEN);
    auto buconv5_end = std::chrono::high_resolution_clock::now();

    auto buconv4_begin = std::chrono::high_resolution_clock::now();
    do_dB_conv(ptr_dconv4, ptr_conv4b, CNN1D_CONV_4_F, CNN1D_CONV_4_F, CNN1D_CONV_4_W, learning_rate, WIDE_LEN);
    auto buconv4_end = std::chrono::high_resolution_clock::now();
    
    auto buconv3_begin = std::chrono::high_resolution_clock::now();
    do_dB_conv(ptr_dconv3, ptr_conv3b, CNN1D_CONV_3_F, CNN1D_CONV_3_F, CNN1D_CONV_3_W, learning_rate, WIDE_LEN);
    auto buconv3_end = std::chrono::high_resolution_clock::now();
    
    auto buconv2_begin = std::chrono::high_resolution_clock::now();
    do_dB_conv(ptr_dconv2, ptr_conv2b, CNN1D_CONV_2_F, CNN1D_CONV_2_F, CNN1D_CONV_2_W, learning_rate, WIDE_LEN);
    auto buconv2_end = std::chrono::high_resolution_clock::now();
    
    auto buconv1_begin = std::chrono::high_resolution_clock::now();
    //do_dB_conv(ptr_dconv1, ptr_conv1b, CNN2_CONV_1_F, CNN2_CONV_1_F, CNN2_CONV_1_W, learning_rate, WIDE_LEN);
    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_create(&(dbconv1_threads[i]), NULL, worker_do_dB_SGD_conv, (void *)dbconv1_data[i]);
    }
            
    for(unsigned i = 0; i < threading_factor; ++i)
    {
        pthread_join(dbconv1_threads[i], NULL);
    }
    auto buconv1_end = std::chrono::high_resolution_clock::now();

    OCL_CHECK(err, q.finish());
    
    auto bu_end = std::chrono::high_resolution_clock::now();
    auto bw_end = std::chrono::high_resolution_clock::now();
    auto all_end = std::chrono::high_resolution_clock::now();

    #ifdef PRINT_DEBUG
        out_file << std::endl << "################################## FW ##################################" << std::endl;
        out_file << std::endl << "################################## CONV FW 1 ##################################" << std::endl;
        print_output_wide(ptr_mp1, CNN1D_CONV_2_C_im2col, CNN1D_CONV_2_W, WIDE_LEN, out_file);
        print_vector(ptr_conv1_ix, MAX_ARRAY_SIZE, out_file);
        out_file << std::endl << "################################## CONV FW 2 ##################################" << std::endl;
        print_output_wide(ptr_mp2, CNN1D_CONV_3_C_im2col, CNN1D_CONV_3_W, WIDE_LEN, out_file);
        print_vector(ptr_conv2_ix, MAX_ARRAY_SIZE, out_file);
        out_file << std::endl << "################################## CONV FW 3 ##################################" << std::endl;
        print_output_wide(ptr_conv3, CNN1D_CONV_4_C_im2col, CNN1D_CONV_3_W, WIDE_LEN, out_file);
        print_vector(ptr_conv3_ix, MAX_ARRAY_SIZE, out_file);
        out_file << std::endl << "Reverse im2col: " << std::endl;
        print_output(conv3_flatten, CNN1D_CONV_3_F, CNN1D_CONV_3_W, out_file);
        out_file << std::endl << "################################## CONV FW 4 ##################################" << std::endl;
        print_output_wide(ptr_conv4, CNN1D_CONV_5_C_im2col, CNN1D_CONV_4_W, WIDE_LEN, out_file);
        print_vector(ptr_conv4_ix, MAX_ARRAY_SIZE, out_file);
        out_file << std::endl << "Reverse im2col: " << std::endl;
        print_output(conv4_flatten, CNN1D_CONV_4_F, CNN1D_CONV_4_W, out_file);
        out_file << std::endl << "################################## CONV FW 5 ##################################" << std::endl;
        print_output_wide(ptr_conv5, CNN1D_CONV_5_F, CNN1D_CONV_5_W, WIDE_LEN, out_file);
        print_vector(ptr_conv5_ix, MAX_ARRAY_SIZE, out_file);
        out_file << std::endl << "################################## FC FW 1 ##################################" << std::endl;        
        printFlattenedMatrix(ptr_fc1, CNN1D_FC_1_F, CNN1D_FC_1_F_widened, out_file);
        out_file << std::endl << "################################## SOFTMAX ##################################" << std::endl;
        print_output_wide(ptr_softmax, 1, CNN1D_FC_1_F_widened, WIDE_LEN, out_file);
        out_file << std::endl << "################################## END FW ##################################" << std::endl;
        
        out_file << std::endl << "################################## dX ##################################" << std::endl;
        out_file << std::endl << "################################## dX FC2 ##################################" << std::endl;
        print_output_wide(ptr_dfc1, 1, CNN1D_FC_1_C, WIDE_LEN, out_file);
        out_file << std::endl << "################################## dX CONV5 (ReLU) ##################################" << std::endl;
        print_output_wide(ptr_dfc1, CNN1D_CONV_5_F, CNN1D_CONV_5_W, WIDE_LEN, out_file);
        out_file << std::endl << "################################## dX CONV5 (Conv) ##################################" << std::endl;
        print_output_wide(ptr_dconv4_relu, CNN1D_CONV_5_F, CNN1D_CONV_5_W, WIDE_LEN, out_file);
        out_file << std::endl << "################################## dX CONV4 (ReLU) ##################################" << std::endl;
        print_output_wide(ptr_dconv4_relu, CNN1D_CONV_5_F, CNN1D_CONV_5_W, WIDE_LEN, out_file);
        out_file << std::endl << "################################## dX CONV4 (Conv) ##################################" << std::endl;
        print_output_wide(ptr_dconv3_relu, CNN1D_CONV_4_F, CNN1D_CONV_4_W, WIDE_LEN, out_file);
        out_file << std::endl << "################################## dX CONV3 (ReLU) ##################################" << std::endl;
        print_output_wide(ptr_dconv3_relu, CNN1D_CONV_4_F, CNN1D_CONV_4_W, WIDE_LEN, out_file);
        out_file << std::endl << "################################## dX CONV3 (Conv) ##################################" << std::endl;
        print_output_wide(ptr_dmp2, CNN1D_CONV_3_F, CNN1D_CONV_3_W, WIDE_LEN, out_file);
        out_file << std::endl << "################################## dX CONV2 (ReLU + MP) ##################################" << std::endl;
        print_output_wide(ptr_dconv2, CNN1D_CONV_3_F, CNN1D_CONV_2_W, WIDE_LEN, out_file);
        out_file << std::endl << "################################## dX CONV2 (Conv) ##################################" << std::endl;
        print_output_wide(ptr_dmp1, CNN1D_CONV_3_C_im2col, CNN1D_CONV_2_W, WIDE_LEN, out_file);
        out_file << std::endl << "################################## dX CONV1 (ReLU + MP) ##################################" << std::endl;
        print_output_wide(ptr_dconv1, CNN1D_CONV_1_F, CNN1D_CONV_1_W, WIDE_LEN, out_file);
        out_file << std::endl << "################################## END dX ##################################" << std::endl;

        out_file << std::endl << "################################## DW ##################################" << std::endl;
        out_file << std::endl << "################################## dW FC1 ##################################" << std::endl;
        print_output_wide(ptr_dwfc1, 512, 16, WIDE_LEN, out_file);
        out_file << std::endl << "################################## dW CONV 5 ##################################" << std::endl;
        print_output_wide(ptr_dwconv5, CNN1D_CONV_5_C_im2col, CNN1D_CONV_5_F, WIDE_LEN, out_file);
        out_file << std::endl << "################################## dW CONV 4 ##################################" << std::endl;
        print_output_wide(ptr_dwconv4, CNN1D_CONV_4_C_im2col, CNN1D_CONV_4_F, WIDE_LEN, out_file);
        out_file << std::endl << "################################## dW CONV 3 ##################################" << std::endl;
        print_output_wide(ptr_dwconv3, CNN1D_CONV_3_C_im2col, CNN1D_CONV_3_F, WIDE_LEN, out_file);
        out_file << std::endl << "################################## dW CONV 2 ##################################" << std::endl;
        print_output_wide(ptr_dwconv2, CNN1D_CONV_2_C_im2col, CNN1D_CONV_2_F, WIDE_LEN, out_file);
        out_file << std::endl << "################################## dW CONV 1 ##################################" << std::endl;
        print_output_wide(ptr_dwconv1, CNN1D_CONV_1_C_im2col_padded, CNN1D_CONV_1_F, WIDE_LEN, out_file);

        out_file << std::endl << "################################## WU FC1 ##################################" << std::endl;
        print_output_wide(ptr_fc1w, CNN1D_FC_1_C, CNN1D_FC_1_F_widened, WIDE_LEN, out_file);
        out_file << std::endl << "################################## WU CONV 5 ##################################" << std::endl;
        print_output_wide(ptr_conv5w_tr, CNN1D_CONV_5_C_im2col, CNN1D_CONV_5_F, WIDE_LEN, out_file);
        out_file << std::endl << "################################## WU CONV 5 TR ##################################" << std::endl;
        print_output_wide(ptr_conv5w, CNN1D_CONV_5_F, CNN1D_CONV_5_C_im2col, WIDE_LEN, out_file);
        out_file << std::endl << "################################## WU CONV 4 ##################################" << std::endl;
        print_output_wide(ptr_conv4w_tr, CNN1D_CONV_4_C_im2col, CNN1D_CONV_4_F, WIDE_LEN, out_file);
        out_file << std::endl << "################################## WU CONV 4 TR ##################################" << std::endl;
        print_output_wide(ptr_conv4w, CNN1D_CONV_4_F, CNN1D_CONV_4_C_im2col, WIDE_LEN, out_file);
        out_file << std::endl << "################################## WU CONV 3 ##################################" << std::endl;
        print_output_wide(ptr_conv3w_tr, CNN1D_CONV_3_C_im2col, CNN1D_CONV_3_F, WIDE_LEN, out_file);
        out_file << std::endl << "################################## WU CONV 3 TR ##################################" << std::endl;
        print_output_wide(ptr_conv3w, CNN1D_CONV_3_F, CNN1D_CONV_3_C_im2col, WIDE_LEN, out_file);
        out_file << std::endl << "################################## WU CONV 2 ##################################" << std::endl;
        print_output_wide(ptr_conv2w_tr, CNN1D_CONV_2_C_im2col, CNN1D_CONV_2_F, WIDE_LEN, out_file);
        out_file << std::endl << "################################## WU CONV 2 TR ##################################" << std::endl;
        print_output_wide(ptr_conv2w, CNN1D_CONV_2_F, CNN1D_CONV_2_C_im2col, WIDE_LEN, out_file);
        out_file << std::endl << "################################## WU CONV 1 ##################################" << std::endl;
        print_output_wide(ptr_conv1w_tr, CNN1D_CONV_1_C_im2col, CNN1D_CONV_1_F, WIDE_LEN, out_file);
        out_file << std::endl << "################################## WU CONV 1 TR ##################################" << std::endl;
        print_output_wide(ptr_conv1w, CNN1D_CONV_1_F, CNN1D_CONV_1_C_im2col_padded, WIDE_LEN, out_file);
        out_file << std::endl << "##################################      BU      ##################################" << std::endl;
        out_file << "FC1 b" << std::endl;
        print_vector(ptr_fc1b, CNN1D_FC_1_F_widened, out_file);
        out_file << "Conv 5 b" << std::endl;
        print_vector(ptr_conv5b, CNN1D_CONV_5_F, out_file);
        out_file << "Conv 4 b" << std::endl;
        print_vector(ptr_conv4b, CNN1D_CONV_4_F, out_file);
        out_file << "Conv 3 b" << std::endl;
        print_vector(ptr_conv3b, CNN1D_CONV_3_F, out_file);
        out_file << "Conv 2 b" << std::endl;
        print_vector(ptr_conv2b, CNN1D_CONV_2_F, out_file);
        out_file << "Conv 1 b" << std::endl;
        print_vector(ptr_conv1b, CNN1D_CONV_1_F, out_file);

    #endif

    std::cout << "End process" << std::endl;

    std::chrono::duration<double> duration_fw = fw_end - fw_begin;
    std::chrono::duration<double> duration_dx = dx_end - dx_begin;
    std::chrono::duration<double> duration_dw = dw_end - dw_begin;
    std::chrono::duration<double> duration_softmax = softmax_end - softmax_begin;
    std::chrono::duration<double> duration_wu = wu_end - wu_begin;
    std::chrono::duration<double> duration_bu = bu_end - bu_begin;
    std::chrono::duration<double> duration_all = all_end - all_begin;
    printf("- FW pass         : %10.4f ms\n",    duration_fw.count() * 1000);
    printf("- Softmax         : %10.4f ms\n",    duration_softmax.count() * 1000);
    printf("- dX         : %10.4f ms\n",    duration_dx.count() * 1000);
    printf("- dW         : %10.4f ms\n",    duration_dw.count() * 1000);
    printf("- WU         : %10.4f ms\n",    duration_wu.count() * 1000);
    printf("- db-bu         : %10.4f ms\n",    duration_bu.count() * 1000);
    printf("- ALL         : %10.4f ms\n",    duration_all.count() * 1000);

    std::cout << "FW" << std::endl;
    std::chrono::duration<double> duration_fwconv1 = conv1_fw_end - conv1_fw_begin;
    std::chrono::duration<double> duration_fwconv2 = conv2_fw_end - conv2_fw_begin;
    std::chrono::duration<double> duration_fwconv3 = conv3_fw_end - conv3_fw_begin;
    std::chrono::duration<double> duration_fwconv4 = conv4_fw_end - conv4_fw_begin;
    std::chrono::duration<double> duration_fwconv5 = conv5_fw_end - conv5_fw_begin;
    std::chrono::duration<double> duration_fwfc1 = fc1_fw_end - fc1_fw_begin;
    printf("- FW Conv 1 (FPGA)         : %10.4f ms\n",    duration_fwconv1.count() * 1000);
    printf("- FW Conv 2 (FPGA)         : %10.4f ms\n",    duration_fwconv2.count() * 1000);
    printf("- FW Conv 3 (FPGA)         : %10.4f ms\n",    duration_fwconv3.count() * 1000);
    printf("- FW Conv 4 (FPGA)         : %10.4f ms\n",    duration_fwconv4.count() * 1000);
    printf("- FW Conv 5 (FPGA)         : %10.4f ms\n",    duration_fwconv5.count() * 1000);
    printf("- FW FC1 (FPGA)         : %10.4f ms\n",    duration_fwfc1.count() * 1000);

    std::cout << "CPU" << std::endl;
    std::chrono::duration<double> duration_rev_im2col_conv1 = conv1_reverse_im2col_end - conv1_reverse_im2col_begin;
    std::chrono::duration<double> duration_rev_im2col_conv2 = conv2_reverse_im2col_end - conv2_reverse_im2col_begin;
    std::chrono::duration<double> duration_rev_im2col_conv3 = conv3_reverse_im2col_end - conv3_reverse_im2col_begin;
    std::chrono::duration<double> duration_rev_im2col_conv4 = conv4_reverse_im2col_end - conv4_reverse_im2col_begin;
    printf("- Conv 1 im2col reverse (CPU)         : %10.4f ms\n",    duration_rev_im2col_conv1.count() * 1000);
    printf("- Conv 2 im2col reverse (CPU)         : %10.4f ms\n",    duration_rev_im2col_conv2.count() * 1000);
    printf("- Conv 3 im2col reverse (CPU)         : %10.4f ms\n",    duration_rev_im2col_conv3.count() * 1000);
    printf("- Conv 4 im2col reverse (CPU)         : %10.4f ms\n",    duration_rev_im2col_conv4.count() * 1000);

    std::cout << "DX" << std::endl;
    std::chrono::duration<double> duration_dxfc1 = dx_fc1_end - dx_fc1_begin;
    std::chrono::duration<double> duration_dx_conv5_relu = dx_conv5_relu_end - dx_conv5_relu_begin;
    std::chrono::duration<double> duration_dxconv5 = dx_conv5_end - dx_conv5_begin;
    std::chrono::duration<double> duration_dx_conv4_relu = dx_conv4_relu_end - dx_conv4_relu_begin;
    std::chrono::duration<double> duration_dxconv4 = dx_conv4_end - dx_conv4_begin;
    std::chrono::duration<double> duration_dx_conv3_relu = dx_conv3_relu_end - dx_conv3_relu_begin;
    std::chrono::duration<double> duration_dxconv3 = dx_conv3_end - dx_conv3_begin;
    std::chrono::duration<double> duration_dx_conv2_relu = dx_conv2_mp_relu_end - dx_conv2_mp_relu_begin;
    std::chrono::duration<double> duration_dxconv2 = dx_conv2_end - dx_conv2_begin;
    std::chrono::duration<double> duration_dx_conv1_relu = dx_conv1_end - dx_conv1_begin;
    printf("- dX FC 1 (FPGA)         : %10.4f ms\n",    duration_dxfc1.count() * 1000);
    printf("- dX Conv 5 ReLU (CPU)         : %10.4f ms\n",    duration_dx_conv5_relu.count() * 1000);
    printf("- dX Conv 5 Conv (FPGA)         : %10.4f ms\n",    duration_dxconv5.count() * 1000);
    printf("- dX Conv 4 ReLU (CPU #4)         : %10.4f ms\n",    duration_dx_conv4_relu.count() * 1000);
    printf("- dX Conv 4 Conv (FPGA)         : %10.4f ms\n",    duration_dxconv4.count() * 1000);
    printf("- dX Conv 3 ReLU (CPU #4)         : %10.4f ms\n",    duration_dx_conv3_relu.count() * 1000);
    printf("- dX Conv 3 Conv (FPGA)         : %10.4f ms\n",    duration_dxconv3.count() * 1000);
    printf("- dX Conv 2 ReLU + MP (CPU #4)         : %10.4f ms\n",    duration_dx_conv2_relu.count() * 1000);
    printf("- dX Conv 2 Conv (FPGA)         : %10.4f ms\n",    duration_dxconv2.count() * 1000);
    printf("- dX Conv 1 ReLU + MP (CPU #4)         : %10.4f ms\n",    duration_dx_conv1_relu.count() * 1000);

    std::cout << "dW" << std::endl;
    std::chrono::duration<double> duration_dwfc1 = dw_fc1_end - dw_fc1_begin;
    std::chrono::duration<double> duration_dwconv5 = dw_conv5_end - dw_conv5_begin;
    std::chrono::duration<double> duration_dwconv4 = dw_conv4_end - dw_conv4_begin;
    std::chrono::duration<double> duration_dwconv3 = dw_conv3_end - dw_conv3_begin;
    std::chrono::duration<double> duration_dwconv2 = dw_conv2_end - dw_conv2_begin;
    std::chrono::duration<double> duration_dwconv1 = dw_conv1_end - dw_conv1_begin;
    printf("- dW FC1 (FPGA)         : %10.4f ms\n",    duration_dwfc1.count() * 1000);
    printf("- dW Conv 5 (FPGA)         : %10.4f ms\n",    duration_dwconv5.count() * 1000);
    printf("- dW Conv 4 (FPGA)         : %10.4f ms\n",    duration_dwconv4.count() * 1000);
    printf("- dW Conv 3 (FPGA)         : %10.4f ms\n",    duration_dwconv3.count() * 1000);
    printf("- dW Conv 2 (FPGA)         : %10.4f ms\n",    duration_dwconv2.count() * 1000);
    printf("- dW Conv 1 (FPGA)         : %10.4f ms\n",    duration_dwconv1.count() * 1000);

    std::cout << "WU" << std::endl;
    std::chrono::duration<double> duration_wufc1 = wufc1_end - wufc1_begin;
    std::chrono::duration<double> duration_wuconv5 = wuconv5_end - wuconv5_begin;
    std::chrono::duration<double> duration_wuconv4 = wuconv4_end - wuconv4_begin;
    std::chrono::duration<double> duration_wuconv3 = wuconv3_end - wuconv3_begin;
    std::chrono::duration<double> duration_wuconv2 = wuconv2_end - wuconv2_begin;
    std::chrono::duration<double> duration_wuconv1 = wuconv1_end - wuconv1_begin;

    printf("- WU FC1 (CPU #4)         : %10.4f ms\n",    duration_wufc1.count() * 1000);
    printf("- WU CONV5 (CPU #4)         : %10.4f ms\n",    duration_wuconv5.count() * 1000);
    printf("- WU CONV4 (CPU #4)         : %10.4f ms\n",    duration_wuconv4.count() * 1000);
    printf("- WU CONV3 (CPU #4)         : %10.4f ms\n",    duration_wuconv3.count() * 1000);
    printf("- WU CONV2 (CPU #4)         : %10.4f ms\n",    duration_wuconv2.count() * 1000);
    printf("- WU CONV1 (CPU #4)         : %10.4f ms\n",    duration_wuconv1.count() * 1000);

    std::cout << "bU" << std::endl;
    std::chrono::duration<double> duration_dbfc1 = bufc1_end - bufc1_begin;
    std::chrono::duration<double> duration_dbconv5 = buconv5_end - buconv5_begin;
    std::chrono::duration<double> duration_dbconv4 = buconv4_end - buconv4_begin;
    std::chrono::duration<double> duration_dbconv3 = buconv3_end - buconv3_begin;
    std::chrono::duration<double> duration_dbconv2 = buconv2_end - buconv2_begin;
    std::chrono::duration<double> duration_dbconv1 = buconv1_end - buconv1_begin;
    printf("- dB FC1 (CPU #1)         : %10.4f ms\n",    duration_dbfc1.count() * 1000);
    printf("- dB Conv 5 (CPU #1)         : %10.4f ms\n",    duration_dbconv5.count() * 1000);
    printf("- dB Conv 4 (CPU #1)         : %10.4f ms\n",    duration_dbconv4.count() * 1000);
    printf("- dB Conv 3 (CPU #1)         : %10.4f ms\n",    duration_dbconv3.count() * 1000);
    printf("- dB Conv 2 (CPU #1)         : %10.4f ms\n",    duration_dbconv2.count() * 1000);
    printf("- dB Conv 1 (CPU #4)         : %10.4f ms\n",    duration_dbconv1.count() * 1000);

    std::cout << "Other CPU" << std::endl;
    std::chrono::duration<double> duration_conv5_transpose = conv5_tr_end - conv5_tr_begin;
    std::chrono::duration<double> duration_conv4_transpose = conv4_tr_end - conv4_tr_begin;
    std::chrono::duration<double> duration_conv3_transpose = conv3_tr_end - conv3_tr_begin;
    std::chrono::duration<double> duration_conv2_transpose = conv2_tr_end - conv2_tr_begin;
    std::chrono::duration<double> duration_conv1_transpose = conv1_tr_end - conv1_tr_begin;
    printf("- CONV5 Weight transpose (CPU #1)         : %10.4f ms\n",    duration_conv5_transpose.count() * 1000);
    printf("- CONV4 Weight transpose (CPU #1)         : %10.4f ms\n",    duration_conv4_transpose.count() * 1000);
    printf("- CONV3 Weight transpose (CPU #1)         : %10.4f ms\n",    duration_conv3_transpose.count() * 1000);
    printf("- CONV2 Weight transpose (CPU #1)         : %10.4f ms\n",    duration_conv2_transpose.count() * 1000);
    printf("- CONV1 Weight transpose (CPU #1)         : %10.4f ms\n",    duration_conv1_transpose.count() * 1000);


    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_input, ptr_input));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv1w, ptr_conv1w));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv2w, ptr_conv2w));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv3w, ptr_conv3w));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv4w, ptr_conv4w));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv5w, ptr_conv5w));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_fc1w, ptr_fc1w));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv1w_tr, ptr_conv1w_tr));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv2w_tr, ptr_conv2w_tr));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv3w_tr, ptr_conv3w_tr));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv4w_tr, ptr_conv4w_tr));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv5w_tr, ptr_conv5w_tr));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv1b, ptr_conv1b));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv2b, ptr_conv2b));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv3b, ptr_conv3b));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv4b, ptr_conv4b));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv5b, ptr_conv5b));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_fc1b, ptr_fc1b));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv1_ix, ptr_conv1_ix));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv2_ix, ptr_conv2_ix));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv3_ix, ptr_conv3_ix));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv4_ix, ptr_conv4_ix));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv5_ix, ptr_conv5_ix));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_fc1_ix, ptr_fc1_ix));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_mp1, ptr_mp1));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_mp2, ptr_mp2));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv3, ptr_conv3));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv4, ptr_conv4));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_conv5, ptr_conv5));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_fc1, ptr_fc1));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_softmax, ptr_softmax));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dfc1, ptr_dfc1));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dconv4, ptr_dconv4));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dconv4_relu, ptr_dconv4_relu));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dconv3, ptr_dconv3));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dconv3_relu, ptr_dconv3_relu));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dconv2, ptr_dconv2));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dmp2, ptr_dmp2));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dmp1, ptr_dmp1));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dconv1, ptr_dconv1));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dwconv1, ptr_dwconv1));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dwconv2, ptr_dwconv2));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dwconv3, ptr_dwconv3));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dwconv4, ptr_dwconv4));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dwconv5, ptr_dwconv5));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_dwfc1, ptr_dwfc1));

    OCL_CHECK(err, err = q.finish());

    delete[] mp1_flatten;
    delete[] conv2_flatten;
    delete[] conv3_flatten;
    delete[] conv4_flatten;

    //=======================================================================================================================

    return 0;

}

