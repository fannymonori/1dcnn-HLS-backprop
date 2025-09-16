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

#include <CL/cl2.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <chrono>
#include <pthread.h>
#include <string> 

#include "ap_fixed.h"
#include "hls_vector.h"
#include "hls_math.h"

#include "types.hpp"

#define M 100
#define N 6
#define D_out 3

typedef ap_fixed<32, 16> math_type;

template <typename T, typename T2>
T CrossEntropyLoss(T *pred, T2 *gt, int n_) {
    T sum = 0;
    for (int i = 0; i < n_; i++) {
        //sum += gt(0, i) * std::log(pred(0, i));
        //sum += gt(0, i) * hls::log((ap_fixed<PRECISION, INTPART>)pred(0, i));
        //sum += gt(0, i) * ap_fixed<32, 16>(hls::log((float)pred(0, i)));
        //sum += gt[i] * ap_fixed<32, 16>(hls::log((float)pred[i]));
        sum += gt[i] * ap_fixed<32, 16>(std::log((float)pred[i]));
    }
    return -1 * sum;
}

static const int DATA_SIZE = 4096;

static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <xclbin> <data_folder>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string xclbinFilename = argv[1];
    std::string pathToData = argv[2];


    ///////////////////////////////////////////////////////////// SET UP DEVICE
    std::vector<cl::Device> devices;
    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    cl::Program program;
    std::vector<cl::Platform> platforms;
    bool found_device = false;
    unsigned iter_count = 0;

    cl::Platform::get(&platforms);
    for (size_t i = 0; (i < platforms.size()) & (found_device == false); i++) {
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if (platformName == "Xilinx") {
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
            if (devices.size()) {
                found_device = true;
                break;
            }
        }
    }
    if (found_device == false) {
        std::cout << "Error: Unable to find Target Device " << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Found platform" << std::endl;

    std::cout << "INFO: Reading " << xclbinFilename << std::endl;
    FILE* fp;
    if ((fp = fopen(xclbinFilename.c_str(), "r")) == nullptr) {
        printf("ERROR: %s xclbin not available please build\n", xclbinFilename.c_str());
        exit(EXIT_FAILURE);
    }

    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    char* buf = new char[nb];
    bin_file.read(buf, nb);

    cl::Kernel krnl_1dcnn;

    cl::Program::Binaries bins;
    bins.push_back({buf, nb});
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        }
        else {
            std::cout << "Device[" << i << "]: program successful!\n";
            
            OCL_CHECK(err, krnl_1dcnn = cl::Kernel(program, "dcnn1d_top_orig", &err));
            
            valid_device = true;
            break;
        }
    }
    
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }


    std::vector<std::vector<int>> labels;
    std::vector<std::vector<HLSNN_DataType>> data;

    std::string line, s;

    std::string data_file = "./Data/train_samples_wine.txt";
    std::ifstream sample_file(data_file);
    if (sample_file.is_open()){
    	int count = 0;

    	while (std::getline(sample_file, line))
    	{
    		std::istringstream ss(line);

    		if(count % 2 == 0){
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

        std::cout << "Data read" << std::endl;
    }
	else{
		std::cout << "Could not open file! " << data_file << std::endl;
	}

    ///////////////////////////////////////////////////////////// SET UP BUFFERS

	int data_size = data.size();

    OCL_CHECK(err, cl::Buffer buffer_input_(context, CL_MEM_READ_ONLY, (M*N) * sizeof(HLSNN_DataType), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_result_(context, CL_MEM_READ_ONLY, (D_out) * sizeof(HLSNN_DataType), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_label_(context, CL_MEM_READ_WRITE, (D_out) * sizeof(int), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_finished_training_(context, CL_MEM_READ_WRITE, (10) * sizeof(bool), NULL, &err));

    HLSNN_DataType *ptr_input, *ptr_result;
    bool *ptr_finished;
    int *ptr_label;
    OCL_CHECK(err, ptr_input = (HLSNN_DataType*)q.enqueueMapBuffer(buffer_input_, CL_TRUE, CL_MAP_WRITE, 0, (M*N) * sizeof(HLSNN_DataType), NULL, NULL, &err));
    OCL_CHECK(err, ptr_result = (HLSNN_DataType*)q.enqueueMapBuffer(buffer_result_, CL_TRUE, CL_MAP_WRITE, 0, (D_out) * sizeof(HLSNN_DataType), NULL, NULL, &err));
    OCL_CHECK(err, ptr_label = (int*)q.enqueueMapBuffer(buffer_label_, CL_TRUE, CL_MAP_WRITE, 0, (D_out) * sizeof(int), NULL, NULL, &err));
    OCL_CHECK(err, ptr_finished = (bool*)q.enqueueMapBuffer(buffer_finished_training_, CL_TRUE, CL_MAP_WRITE, 0, (10) * sizeof(bool), NULL, NULL, &err));

    int narg = 0;
    OCL_CHECK(err, err = krnl_1dcnn.setArg(narg++, buffer_input_));
    OCL_CHECK(err, err = krnl_1dcnn.setArg(narg++, buffer_result_));
    OCL_CHECK(err, err = krnl_1dcnn.setArg(narg++, buffer_label_));
    OCL_CHECK(err, err = krnl_1dcnn.setArg(narg++, true));
    OCL_CHECK(err, err = krnl_1dcnn.setArg(narg++, buffer_finished_training_));
    OCL_CHECK(err, err = krnl_1dcnn.setArg(narg++, true));

    ///////////////////////////////////////////////////////////// RUN KERNEL

    for(int d = 0; d < MAX_EPOCH; d++){
        std::cout << "Sample number #" << d << std::endl; 

        if(d == 0){
            OCL_CHECK(err, err = krnl_1dcnn.setArg(5, true));
        }
        else{
            OCL_CHECK(err, err = krnl_1dcnn.setArg(5, false));
        }

        for(unsigned i = 0; i < 10; i++){
            ptr_finished[i] = false;
        }

        //std::cout << "Input: " << std::endl;
        for(unsigned i = 0; i < M*N; i++){
            ptr_input[i] = data[d][i];
        }

        //std::cout << "Label: " << std::endl;
        for(unsigned i = 0; i < D_out; i++){
            //std::cout << labels[d][i] << " ";
            ptr_label[i] = labels[d][i];
        }
        //std::cout << std::endl;

        //std::cout << "Result: " << std::endl;
        for(unsigned i = 0; i < D_out; i++){
            ptr_result[i] = 0.0;
        }

        auto fpga_begin = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input_, buffer_label_, buffer_result_, buffer_finished_training_}, 0 ));
        OCL_CHECK(err, err = q.enqueueTask(krnl_1dcnn));
        OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_input_, buffer_label_, buffer_result_, buffer_finished_training_}, CL_MIGRATE_MEM_OBJECT_HOST));
        OCL_CHECK(err, q.finish());
        auto fpga_end = std::chrono::high_resolution_clock::now();

        std::cout << "Result: " << std::endl;
        for(unsigned i = 0; i < D_out; i++){
            std::cout << ptr_result[i] << " ";
        }
        std::cout << std::endl;

        double loss = CrossEntropyLoss(ptr_result, ptr_label, 3);
        std::cout << "Loss from PL inference: " << loss << " Training finished: " << ptr_finished[0] << std::endl;

        std::chrono::duration<double> fpga_duration = fpga_end - fpga_begin;
        printf("- FPGA Time   : %10.4f ms\n",    fpga_duration.count() * 1000.0);
    }

    return 0;
}
