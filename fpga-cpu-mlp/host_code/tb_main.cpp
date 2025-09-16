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

#include <CL/cl2.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <stdlib.h>
#include <chrono>
#include <pthread.h>
#include <chrono>
#include <thread>
#include <thread>
#include <iomanip>
#include "cnpy.h"

#include "ap_fixed.h"
#include "hls_vector.h"
#include "hls_math.h"

#include "utils.hpp"
#include "types.hpp"
#include "parameter_update.hpp"
#include "activation.hpp"

using namespace std::this_thread; // sleep_for, sleep_until
using namespace std::chrono; 

const float beta_1=0.9;
const float beta_2=0.999;


const float inv_beta_1=0.1;
const float inv_beta_2=0.001;

const float epsilon=0.000001;

const float learning_rate=0.001;

typedef float math_type;

//Customized buffer allocation for 4K boundary alignment
template <typename T>
struct aligned_allocator
{
  using value_type = T;
  T* allocate(std::size_t num)
  {
    void* ptr = nullptr;
    if (posix_memalign(&ptr,4096,num*sizeof(T)))
      throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t num)
  {
    free(p);
  }
};

//static const int DATA_SIZE = 4096;

static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

#include "tb_all_fpga.hpp"
//#include "tb_cpu_fpga.hpp"


int main(int argc, char* argv[]) {

    std::cout << "Start of program" << std::endl;
    
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <xclbin> <data_folder>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string xclbinFilename = argv[1];
    std::string pathToData = argv[2];

    int return_code = 0;

    std::vector<cl::Device> devices;
    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    cl::Program program;
    std::vector<cl::Platform> platforms;
    bool found_device = false;

    // GET PLATFORM
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
    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    char* buf = new char[nb];
    bin_file.read(buf, nb);

    cl::Program::Binaries bins;
    bins.push_back({buf, nb});
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        //cl::Program program(context, {device}, bins, nullptr, &err);
        program = cl::Program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        }
        else {
            std::cout << "Device[" << i << "]: program successful!\n";
            
            valid_device = true;
            break;
        }
    }

    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
    else{
        //working benchmark
        return_code = run_all_on_FPGA(xclbinFilename, pathToData, context, q, program);
        //return_code = run_on_CPU_FPGA(xclbinFilename, pathToData, context, q, program);
    }

    std::cout << "Finished NN training testbench" << std::endl;

    return 0;
}