#include "types.hpp"
#define max_m_ 3500
#define max_f_ 96
#define max_c_ 96
#define max_b_ 96
#define max_k_ 4

/**
* Run computations on the CPU for comparison.
* These functions use simple array computations
*/
void run_conv_on_CPU(LayerConfig &lcfg, float &fw_time, float &dx_time, float &dw_time, float &mp, float &mp_bp){

    unsigned signal_length = lcfg.signal_length;
    unsigned kernel_ = lcfg.kernel_size;
    unsigned ch_ = lcfg.input_features;
    unsigned f_ = lcfg.output_features;

    unsigned ifm_size = (signal_length + 2) * ch_;
    unsigned ofm_size = (signal_length + 2) * f_;
    unsigned w_size = kernel_ * f_ * ch_;
    unsigned b_size = f_;

    float ifm_array[max_m_][max_c_];
    float w_array[max_k_][max_c_][max_f_];
    float w_array_flip[max_k_][max_f_][max_c_];
    float b_array[max_f_];
	float ofm_array[max_m_][max_f_];
    float ofm_array2[max_m_][max_f_];
    float ofm_array3[max_m_][max_f_];

    fill_conv_arrays_random(ifm_array, w_array, w_array_flip, b_array, ofm_array, ofm_array2, ofm_array3, signal_length, 2, kernel_, ch_, f_);

    auto cpu_fw_begin = std::chrono::high_resolution_clock::now();
    CONV1D_ARRAY(ifm_array, w_array, b_array, ofm_array, signal_length, 0, kernel_, ch_, f_);
    auto cpu_fw_end = std::chrono::high_resolution_clock::now();

    auto cpu_dx_begin = std::chrono::high_resolution_clock::now();
    CONV1D_ARRAY(ofm_array, w_array_flip, b_array, ifm_array, signal_length, 0, kernel_, f_, ch_);
    auto cpu_dx_end = std::chrono::high_resolution_clock::now();

    auto cpu_dw_begin = std::chrono::high_resolution_clock::now();
    CONV1D_ARRAY_DW(ifm_array, w_array, ofm_array, signal_length, 2, kernel_, ch_, f_);
    auto cpu_dw_end = std::chrono::high_resolution_clock::now();

    auto mp_begin = std::chrono::high_resolution_clock::now();
    MaxPool1D(ofm_array, ofm_array2, signal_length, f_, 1);
    auto mp_end = std::chrono::high_resolution_clock::now();

    auto mp_bp_begin = std::chrono::high_resolution_clock::now();
    MaxPool1D_backprop(ofm_array, ofm_array2, ofm_array3, signal_length, f_, 1);
    auto mp_bp_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> cpu_fw = cpu_fw_end - cpu_fw_begin;
    std::chrono::duration<double> cpu_dx = cpu_dx_end - cpu_dx_begin;
    std::chrono::duration<double> cpu_dw = cpu_dw_end - cpu_dw_begin;
    std::chrono::duration<double> cpu_mp = mp_end - mp_begin;
    std::chrono::duration<double> cpu_mp_bp = mp_bp_end - mp_bp_begin;

    printf("- CPU Time layer FW         : %10.4f ms\n",    cpu_fw.count() * 1000.0);
    printf("- CPU Time layer DX         : %10.4f ms\n",    cpu_dx.count() * 1000.0);
    printf("- CPU Time layer DW         : %10.4f ms\n",    cpu_dw.count() * 1000.0);
    printf("- CPU Time layer MaxPooling         : %10.4f ms\n",    cpu_mp.count() * 1000.0);
    printf("- CPU Time layer MaxPooling BP         : %10.4f ms\n",    cpu_mp_bp.count() * 1000.0);

    fw_time = cpu_fw.count() * 1000.0;
    dx_time = cpu_dx.count() * 1000.0;
    dw_time = cpu_dw.count() * 1000.0;
    mp = cpu_mp.count() * 1000.0;
    mp_bp = cpu_mp_bp.count() * 1000.0;

}

// Run convolutional layer configurations on FPGA and on CPU
int run_multiple_layers_wide(std::string xclbinFilename_, std::string pathToData_, cl::Context &context, cl::CommandQueue &q, cl::Program &program) {
    std::string xclbinFilename = xclbinFilename_;
    std::string pathToData = pathToData_;

    ///////////////////////////////////////////////////////////// SET UP DEVICE
    cl_int err;
    std::vector<LayerConfig>::iterator itr;
    unsigned iter_count = 0;

    std::vector<cl::Kernel> kernel_vector_fw;
    std::vector<cl::Kernel> kernel_vector_dx;
    std::vector<cl::Kernel> kernel_vector_dw;

    std::vector<LayerConfig> layer_configs = {

                /*LayerConfig(LayerType::CONV, 1, 3072, 16, 16, 2), // Large FM, small kernel
                LayerConfig(LayerType::CONV, 1, 3072, 32, 32, 2), // Large FM, medium kernel
                LayerConfig(LayerType::CONV, 1, 3072, 64, 64, 4), // Large FM, large kernel
                LayerConfig(LayerType::CONV, 1, 3072, 96, 96, 4), // Large FM, small kernel*/

                /*LayerConfig(LayerType::CONV, 1, 2048, 16, 16, 2), // Large FM, small kernel
                LayerConfig(LayerType::CONV, 1, 2048, 32, 32, 2), // Large FM, medium kernel
                LayerConfig(LayerType::CONV, 1, 2048, 64, 64, 4), // Large FM, large kernel
                LayerConfig(LayerType::CONV, 1, 2048, 96, 96, 4), // Large FM, small kernel*/

                /*LayerConfig(LayerType::CONV, 1, 1024, 16, 16, 2), // Large FM, small kernel
                LayerConfig(LayerType::CONV, 1, 1024, 32, 32, 2), // Large FM, medium kernel
                LayerConfig(LayerType::CONV, 1, 1024, 64, 64, 4), // Large FM, large kernel
                LayerConfig(LayerType::CONV, 1, 1024, 96, 96, 4), // Large FM, small kernel*/

                /*LayerConfig(LayerType::CONV, 1, 256, 16, 16, 2), // Medium FM, small kernel
                LayerConfig(LayerType::CONV, 1, 256, 32, 32, 2), // Medium FM, medium kernel
                LayerConfig(LayerType::CONV, 1, 256, 64, 64, 4), // Medium FM, large kernel
                LayerConfig(LayerType::CONV, 1, 256, 96, 96, 4), // Large FM, small kernel*/

                LayerConfig(LayerType::CONV, 1, 64, 16, 16, 2), // Small FM, small kernel
                LayerConfig(LayerType::CONV, 1, 64, 32, 32, 2), // Small FM, medium kernel
                LayerConfig(LayerType::CONV, 1, 64, 64, 64, 4), // Small FM, large kernel
                LayerConfig(LayerType::CONV, 1, 64, 96, 96, 4), // Large FM, small kernel
    }; 

    unsigned num_of_metrics = 16;
    unsigned layer_num = 4;
    float results[num_of_metrics][layer_num];

    iter_count = 0;
    for(itr = layer_configs.begin(); itr != layer_configs.end(); itr++){
        cl::Kernel krnl_fw;
        cl::Kernel krnl_dw;
        cl::Kernel krnl_dx;

        OCL_CHECK(err, krnl_fw = cl::Kernel(program, "top_mm_im2col", &err));
        OCL_CHECK(err, krnl_dw = cl::Kernel(program, "top_mm_im2col", &err));
        OCL_CHECK(err, krnl_dx = cl::Kernel(program, "top_mm_im2col", &err));

        kernel_vector_fw.push_back(krnl_fw);
        kernel_vector_dw.push_back(krnl_dw);
        kernel_vector_dx.push_back(krnl_dx);
    }

    unsigned num_of_layers = layer_configs.size();

    // Create OpenCL buffers for all layer types
    std::vector<cl::Buffer> buffer_vector_IFM;
    std::vector<cl::Buffer> buffer_vector_OFM;
    std::vector<cl::Buffer> buffer_vector_W;
    std::vector<cl::Buffer> buffer_vector_B;
    std::vector<cl::Buffer> buffer_vector_ix;
    std::vector<cl::Buffer> buffer_vector_B_grad;
    std::vector<cl::Buffer> buffer_vector_OFM_grad;
    std::vector<cl::Buffer> buffer_vector_IFM_grad;
    std::vector<cl::Buffer> buffer_vector_W_grad;
    std::vector<cl::Buffer> buffer_vector_W_flip;

    // Create pointer handles to buffers
    std::vector<wide_type*> ptr_vector_IFM;
    std::vector<wide_type*> ptr_vector_OFM;
    std::vector<wide_type*> ptr_vector_W;
    std::vector<HLSNN_DataType*> ptr_vector_B;
    std::vector<unsigned*> ptr_vector_ix;
    std::vector<wide_type*> ptr_vector_W_grad;
    std::vector<wide_type*> ptr_vector_OFM_grad;
    std::vector<HLSNN_DataType*> ptr_vector_B_grad;
    std::vector<wide_type*> ptr_vector_IFM_grad;
    std::vector<wide_type*> ptr_vector_W_flip;

    ptr_vector_IFM.resize(num_of_layers);
    ptr_vector_OFM.resize(num_of_layers);
    ptr_vector_B.resize(num_of_layers);
    ptr_vector_ix.resize(num_of_layers);
    ptr_vector_W.resize(num_of_layers);
    ptr_vector_W_grad.resize(num_of_layers); 
    ptr_vector_OFM_grad.resize(num_of_layers);
    ptr_vector_B_grad.resize(num_of_layers);
    ptr_vector_IFM_grad.resize(num_of_layers);
    ptr_vector_W_flip.resize(num_of_layers);

    // Create first and second moment vectors for the ADAM algorithm
    std::vector<float> v_;
    std::vector<float> m_;
    std::vector<float> b_v_;
    std::vector<float> b_m_;

    iter_count = 0;
    for(itr = layer_configs.begin(); itr != layer_configs.end(); itr++){

        unsigned rows_in = (*itr).rows_in;
        unsigned cols_in = (*itr).cols_in;
        unsigned cols_out = (*itr).cols_out;
        unsigned kernel_size_ = (*itr).kernel_size;
        unsigned b_size = cols_in;

        // W x IFM = OFM
        // [F, K*C] x [K*C, W] = [F, W]
        // [rows_in, cols_in] X [cols_in, cols_out] = [rows_in, cols_out] //normal mode output
        // [F, K*C] x [K*C, W] = [K*F, W]
        // [rows_in, cols_in] X [cols_in, cols_out] = [K * rows_in, cols_out] //im2col mode output
        unsigned ifm_size = (kernel_size_*cols_in * (cols_out / WIDE_LEN));
        unsigned w_size = (kernel_size_*rows_in * (cols_in / WIDE_LEN));
        unsigned ofm_size = (kernel_size_*rows_in * (cols_out / WIDE_LEN));

        OCL_CHECK(err, cl::Buffer buffer_ifm_(context, CL_MEM_READ_ONLY, (ifm_size) * sizeof(wide_type), NULL, &err));
        OCL_CHECK(err, cl::Buffer buffer_ifm_grad_(context, CL_MEM_READ_ONLY, (ifm_size) * sizeof(wide_type), NULL, &err));
        OCL_CHECK(err, cl::Buffer buffer_w_(context, CL_MEM_READ_WRITE, (w_size) * sizeof(wide_type), NULL, &err));
        OCL_CHECK(err, cl::Buffer buffer_w_flip_(context, CL_MEM_READ_WRITE, (w_size) * sizeof(wide_type), NULL, &err));
        OCL_CHECK(err, cl::Buffer buffer_w_grad_(context, CL_MEM_READ_WRITE, (w_size) * sizeof(wide_type), NULL, &err));
        OCL_CHECK(err, cl::Buffer buffer_b_(context, CL_MEM_READ_WRITE, (b_size) * sizeof(HLSNN_DataType), NULL, &err));
        OCL_CHECK(err, cl::Buffer buffer_b_grad_(context, CL_MEM_READ_WRITE, (b_size) * sizeof(HLSNN_DataType), NULL, &err));
        OCL_CHECK(err, cl::Buffer buffer_ofm_(context, CL_MEM_READ_WRITE, (ofm_size) * sizeof(wide_type), NULL, &err));
        OCL_CHECK(err, cl::Buffer buffer_ofm_grad_(context, CL_MEM_READ_WRITE, (ofm_size) * sizeof(wide_type), NULL, &err));
        OCL_CHECK(err, cl::Buffer buffer_ix_(context, CL_MEM_READ_WRITE, (ifm_size) * sizeof(unsigned), NULL, &err));

        buffer_vector_IFM.push_back(buffer_ifm_);
        buffer_vector_OFM.push_back(buffer_ofm_);
        buffer_vector_W.push_back(buffer_w_);
        buffer_vector_B.push_back(buffer_b_);
        buffer_vector_ix.push_back(buffer_ix_);
        buffer_vector_B_grad.push_back(buffer_b_grad_);
        buffer_vector_OFM_grad.push_back(buffer_ofm_grad_);
        buffer_vector_IFM_grad.push_back(buffer_ifm_grad_);
        buffer_vector_W_grad.push_back(buffer_w_grad_);
        buffer_vector_W_flip.push_back(buffer_w_flip_);

        int narg = 0;
        OCL_CHECK(err, err = kernel_vector_fw[iter_count].setArg(narg++, buffer_vector_OFM[iter_count])); // output
        OCL_CHECK(err, err = kernel_vector_fw[iter_count].setArg(narg++, buffer_vector_IFM[iter_count])); // input
        OCL_CHECK(err, err = kernel_vector_fw[iter_count].setArg(narg++, buffer_vector_W[iter_count])); // weight
        OCL_CHECK(err, err = kernel_vector_fw[iter_count].setArg(narg++, buffer_vector_B[iter_count])); // bias
        OCL_CHECK(err, err = kernel_vector_fw[iter_count].setArg(narg++, buffer_vector_ix[iter_count])); // indices
        OCL_CHECK(err, err = kernel_vector_fw[iter_count].setArg(narg++, 0));  // mode: 0-fw, 1-transposed, 2-fc dW
        OCL_CHECK(err, err = kernel_vector_fw[iter_count].setArg(narg++, rows_in));
        OCL_CHECK(err, err = kernel_vector_fw[iter_count].setArg(narg++, cols_in));
        OCL_CHECK(err, err = kernel_vector_fw[iter_count].setArg(narg++, cols_out));
        OCL_CHECK(err, err = kernel_vector_fw[iter_count].setArg(narg++, cols_out));
        OCL_CHECK(err, err = kernel_vector_fw[iter_count].setArg(narg++, kernel_size_)); // kernel size for next layer
        OCL_CHECK(err, err = kernel_vector_fw[iter_count].setArg(narg++, (cols_out / WIDE_LEN)));
        OCL_CHECK(err, err = kernel_vector_fw[iter_count].setArg(narg++, true)); // transform output with im2col before saving
        OCL_CHECK(err, err = kernel_vector_fw[iter_count].setArg(narg++, false));
        OCL_CHECK(err, err = kernel_vector_fw[iter_count].setArg(narg++, false));
        OCL_CHECK(err, err = kernel_vector_fw[iter_count].setArg(narg++, false));
        OCL_CHECK(err, err = kernel_vector_fw[iter_count].setArg(narg++, true));
        OCL_CHECK(err, err = kernel_vector_fw[iter_count].setArg(narg++, false));
        OCL_CHECK(err, err = kernel_vector_fw[iter_count].setArg(narg++, false));

        narg = 0;
        OCL_CHECK(err, err = kernel_vector_dx[iter_count].setArg(narg++, buffer_vector_IFM_grad[iter_count])); // input gradient (output is saved here)
        OCL_CHECK(err, err = kernel_vector_dx[iter_count].setArg(narg++, buffer_vector_OFM_grad[iter_count])); // output gradient (this serves as input)
        OCL_CHECK(err, err = kernel_vector_dx[iter_count].setArg(narg++, buffer_vector_W_flip[iter_count])); // flipped weight
        OCL_CHECK(err, err = kernel_vector_dx[iter_count].setArg(narg++, buffer_vector_B[iter_count])); // bias, not used
        OCL_CHECK(err, err = kernel_vector_dx[iter_count].setArg(narg++, buffer_vector_ix[iter_count])); // indices
        OCL_CHECK(err, err = kernel_vector_dx[iter_count].setArg(narg++, 0)); // mode: 0-fw, 1-transposed, 2-fc dW
        OCL_CHECK(err, err = kernel_vector_dx[iter_count].setArg(narg++, cols_in));
        OCL_CHECK(err, err = kernel_vector_dx[iter_count].setArg(narg++, rows_in));
        OCL_CHECK(err, err = kernel_vector_dx[iter_count].setArg(narg++, cols_out));
        OCL_CHECK(err, err = kernel_vector_dx[iter_count].setArg(narg++, cols_out));
        OCL_CHECK(err, err = kernel_vector_dx[iter_count].setArg(narg++, kernel_size_)); // kernel size for this layer
        OCL_CHECK(err, err = kernel_vector_dx[iter_count].setArg(narg++, cols_out));
        OCL_CHECK(err, err = kernel_vector_dx[iter_count].setArg(narg++, false));
        OCL_CHECK(err, err = kernel_vector_dx[iter_count].setArg(narg++, true));  // transform output with col2im before saving
        OCL_CHECK(err, err = kernel_vector_dx[iter_count].setArg(narg++, false));
        OCL_CHECK(err, err = kernel_vector_dx[iter_count].setArg(narg++, false));
        OCL_CHECK(err, err = kernel_vector_dx[iter_count].setArg(narg++, false));
        OCL_CHECK(err, err = kernel_vector_dx[iter_count].setArg(narg++, false));
        OCL_CHECK(err, err = kernel_vector_dx[iter_count].setArg(narg++, false));

        narg = 0;
        OCL_CHECK(err, err = kernel_vector_dw[iter_count].setArg(narg++, buffer_vector_OFM_grad[iter_count]));  // output gradient is read
        OCL_CHECK(err, err = kernel_vector_dw[iter_count].setArg(narg++, buffer_vector_IFM[iter_count])); // input is read
        OCL_CHECK(err, err = kernel_vector_dw[iter_count].setArg(narg++, buffer_vector_W_grad[iter_count])); // weights are written here
        OCL_CHECK(err, err = kernel_vector_dw[iter_count].setArg(narg++, buffer_vector_B[iter_count]));
        OCL_CHECK(err, err = kernel_vector_dw[iter_count].setArg(narg++, buffer_vector_ix[iter_count])); // indices
        OCL_CHECK(err, err = kernel_vector_dw[iter_count].setArg(narg++, 1));
        OCL_CHECK(err, err = kernel_vector_dw[iter_count].setArg(narg++, rows_in));
        OCL_CHECK(err, err = kernel_vector_dw[iter_count].setArg(narg++, cols_in));
        OCL_CHECK(err, err = kernel_vector_dw[iter_count].setArg(narg++, cols_out));
        OCL_CHECK(err, err = kernel_vector_dw[iter_count].setArg(narg++, cols_out));
        OCL_CHECK(err, err = kernel_vector_dw[iter_count].setArg(narg++, kernel_size_));
        OCL_CHECK(err, err = kernel_vector_dw[iter_count].setArg(narg++, (cols_in / WIDE_LEN)));
        OCL_CHECK(err, err = kernel_vector_dw[iter_count].setArg(narg++, false));
        OCL_CHECK(err, err = kernel_vector_dw[iter_count].setArg(narg++, false));
        OCL_CHECK(err, err = kernel_vector_dw[iter_count].setArg(narg++, false));
        OCL_CHECK(err, err = kernel_vector_dw[iter_count].setArg(narg++, false));
        OCL_CHECK(err, err = kernel_vector_dw[iter_count].setArg(narg++, false));
        OCL_CHECK(err, err = kernel_vector_dw[iter_count].setArg(narg++, false));
        OCL_CHECK(err, err = kernel_vector_dw[iter_count].setArg(narg++, false));


        OCL_CHECK(err, ptr_vector_IFM[iter_count] = (wide_type*)q.enqueueMapBuffer(buffer_vector_IFM[iter_count], CL_TRUE, CL_MAP_WRITE, 0, (ifm_size) * sizeof(wide_type), NULL, NULL, &err));
        OCL_CHECK(err, ptr_vector_IFM_grad[iter_count] = (wide_type*)q.enqueueMapBuffer(buffer_vector_IFM_grad[iter_count], CL_TRUE, CL_MAP_WRITE, 0, (ifm_size) * sizeof(wide_type), NULL, NULL, &err));
        OCL_CHECK(err, ptr_vector_W[iter_count] = (wide_type*)q.enqueueMapBuffer(buffer_vector_W[iter_count], CL_TRUE, CL_MAP_WRITE, 0, (w_size) * sizeof(wide_type), NULL, NULL, &err));
        OCL_CHECK(err, ptr_vector_W_flip[iter_count] = (wide_type*)q.enqueueMapBuffer(buffer_vector_W_flip[iter_count], CL_TRUE, CL_MAP_WRITE, 0, (w_size) * sizeof(wide_type), NULL, NULL, &err));
        OCL_CHECK(err, ptr_vector_W_grad[iter_count] = (wide_type*)q.enqueueMapBuffer(buffer_vector_W_grad[iter_count], CL_TRUE, CL_MAP_READ, 0, (w_size) * sizeof(wide_type), NULL, NULL, &err));
        OCL_CHECK(err, ptr_vector_B_grad[iter_count] = (HLSNN_DataType*)q.enqueueMapBuffer(buffer_vector_B_grad[iter_count], CL_TRUE, CL_MAP_WRITE, 0, (b_size) * sizeof(HLSNN_DataType), NULL, NULL, &err));
        OCL_CHECK(err, ptr_vector_B[iter_count] = (HLSNN_DataType*)q.enqueueMapBuffer(buffer_vector_B[iter_count], CL_TRUE, CL_MAP_WRITE, 0, (b_size) * sizeof(HLSNN_DataType), NULL, NULL, &err));
        OCL_CHECK(err, ptr_vector_OFM[iter_count] = (wide_type*)q.enqueueMapBuffer(buffer_vector_OFM[iter_count], CL_TRUE, CL_MAP_READ, 0, (ofm_size) * sizeof(wide_type), NULL, NULL, &err));
        OCL_CHECK(err, ptr_vector_OFM_grad[iter_count] = (wide_type*)q.enqueueMapBuffer(buffer_vector_OFM_grad[iter_count], CL_TRUE, CL_MAP_READ, 0, (ofm_size) * sizeof(wide_type), NULL, NULL, &err));
        OCL_CHECK(err, ptr_vector_ix[iter_count] = (unsigned*)q.enqueueMapBuffer(buffer_vector_ix[iter_count], CL_TRUE, CL_MAP_READ, 0, (ifm_size) * sizeof(unsigned), NULL, NULL, &err));

        iter_count++;
    }

    iter_count = 0;
    for(itr = layer_configs.begin(); itr != layer_configs.end(); itr++){        

        LayerConfig cfg = (*itr);

        // SETUP LAYER DATA
        std::vector<float> result_vector_tmp;

        unsigned rows_in = cfg.rows_in;
        unsigned cols_in = cfg.cols_in;
        unsigned cols_out = cfg.cols_out;
        unsigned kernel_size_ = cfg.kernel_size;

        unsigned ifm_size = cols_in * cols_out;
        unsigned ofm_size = rows_in * cols_out;
        unsigned w_size = rows_in * cols_in;
        unsigned b_size = cols_in;

        unsigned ifm_wide_len = cols_in * (cols_out / WIDE_LEN);
        unsigned ofm_wide_len = rows_in * (cols_out / WIDE_LEN);
        unsigned w_wide_len = rows_in * (cols_in / WIDE_LEN);

        v_.resize(w_size);
        m_.resize(w_size);

        b_v_.resize(b_size);
        b_m_.resize(b_size);

        fill_random(v_.data(), w_size);
        fill_random(m_.data(), w_size);

        fill_random(b_v_.data(), b_size);
        fill_random(b_m_.data(), b_size);

        // Fill those arrays randomly that are used as inputs-only
        fill_random_wide(ptr_vector_IFM[iter_count], ifm_wide_len, WIDE_LEN);
        fill_random_wide(ptr_vector_OFM_grad[iter_count], ofm_wide_len, WIDE_LEN);
        fill_random_wide(ptr_vector_W[iter_count], w_wide_len, WIDE_LEN);
        fill_random_wide(ptr_vector_W_flip[iter_count], w_wide_len, WIDE_LEN);
        fill_random(ptr_vector_B[iter_count], b_size);
        fill_random(ptr_vector_B_grad[iter_count], b_size);

        // Fill those arrays with zeros that are used as outputs
        fill_with_zeros_wide(ptr_vector_OFM[iter_count], ofm_wide_len, WIDE_LEN);
        fill_with_zeros_wide(ptr_vector_W_grad[iter_count], w_wide_len, WIDE_LEN);
        fill_with_zeros_wide(ptr_vector_IFM_grad[iter_count], ifm_wide_len, WIDE_LEN);

        // set up multithreading
        unsigned lr_ = 0.001;
        std::vector<SGD_thread_data_wide*> sgd_data_vector;
        std::vector<ADAM_thread_data_wide*> adam_data_vector;
        std::vector<dB_thread_data*> bias_data_vector;
        std::vector<pthread_t> sgd_threads;
        std::vector<pthread_t> adam_threads;
        std::vector<pthread_t> bias_threads;
        // Multithread execution
        unsigned array_in_4 = int(w_wide_len / 4);
        unsigned bias_array_in_4 = int(rows_in / 4);
        std::cout << "W length: " << w_wide_len << std::endl;
        for(unsigned i = 0; i < 4; ++i)
        {
            unsigned start = i * array_in_4;
            unsigned end = start + array_in_4;
            // THREAD data for SGD
            struct SGD_thread_data_wide *sgd_data = (struct SGD_thread_data_wide *)malloc(sizeof(struct SGD_thread_data_wide));
            sgd_data->w_orig = ptr_vector_W[iter_count];
            sgd_data->w = ptr_vector_W_grad[iter_count];
            sgd_data->LR = lr_;
            sgd_data->length = w_wide_len;
            sgd_data->start = start;
            sgd_data->end = end;
            sgd_data->wide_length = WIDE_LEN;
            pthread_t sgd_thrd;
            sgd_data_vector.push_back(sgd_data);
            sgd_threads.push_back(sgd_thrd);

            // THREAD data for ADAM
            struct ADAM_thread_data_wide *adam_data = (struct ADAM_thread_data_wide *)malloc(sizeof(struct ADAM_thread_data_wide));
            adam_data->w_orig = ptr_vector_W[iter_count];
            adam_data->w = ptr_vector_W_grad[iter_count];
            adam_data->m_array = m_.data();
            adam_data->v_array = v_.data();
            adam_data->LR = lr_;
            adam_data->length = w_wide_len;
            adam_data->start = start;
            adam_data->end = end;
            adam_data->wide_length = WIDE_LEN;
            adam_data->time_step = 1;
            pthread_t adam_thrd;
            adam_data_vector.push_back(adam_data);
            adam_threads.push_back(adam_thrd);


            unsigned bias_start = i * bias_array_in_4;
            unsigned bias_end = bias_start + bias_array_in_4;
            struct dB_thread_data *bias_data = (struct dB_thread_data *)malloc(sizeof(struct dB_thread_data));
            bias_data->ofm = ptr_vector_OFM_grad[iter_count];
            bias_data->b_grad = ptr_vector_B_grad[iter_count];
            bias_data->start = bias_start;
            bias_data->end = bias_end;
            bias_data->C = rows_in;
            bias_data->W = cols_out;
            bias_data->wide_length = WIDE_LEN;
            pthread_t bias_thrd;
            bias_data_vector.push_back(bias_data);
            bias_threads.push_back(bias_thrd);
        }

        // RUN LAYERS

        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({
                    buffer_vector_IFM[iter_count],
                    buffer_vector_W[iter_count],
                    buffer_vector_W_grad[iter_count],
                    buffer_vector_W_flip[iter_count],
                    buffer_vector_B[iter_count],
                    buffer_vector_OFM[iter_count],
                    buffer_vector_OFM_grad[iter_count],
                    buffer_vector_IFM_grad[iter_count]
                    },
                    0 ));

        // Forward pass
        auto begin_FPGA = std::chrono::high_resolution_clock::now();
        auto fpga_begin_fw = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueTask(kernel_vector_fw[iter_count]));
        OCL_CHECK(err, q.finish());
        auto fpga_end_fw = std::chrono::high_resolution_clock::now();

        // dX
        auto begin_backward_pass = std::chrono::high_resolution_clock::now();
        auto fpga_begin_dx = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueTask(kernel_vector_dx[iter_count]));
        OCL_CHECK(err, q.finish());
        auto fpga_end_dx = std::chrono::high_resolution_clock::now();

        // dW
        auto fpga_begin_dw = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueTask(kernel_vector_dw[iter_count]));
        OCL_CHECK(err, q.enqueueMigrateMemObjects({
                    buffer_vector_IFM[iter_count],
                    buffer_vector_W[iter_count],
                    buffer_vector_W_grad[iter_count],
                    buffer_vector_W_flip[iter_count],
                    buffer_vector_B[iter_count],
                    buffer_vector_OFM[iter_count],
                    buffer_vector_OFM_grad[iter_count],
                    buffer_vector_IFM_grad[iter_count]
                    },
                    CL_MIGRATE_MEM_OBJECT_HOST));
        OCL_CHECK(err, q.finish());
        auto fpga_end_dw = std::chrono::high_resolution_clock::now();
        auto end_backward_pass = std::chrono::high_resolution_clock::now();
        auto end_FPGA = std::chrono::high_resolution_clock::now();

        sleep_for(seconds(1));

        auto cpu_begin_adam= std::chrono::high_resolution_clock::now();
        doWU_ADAM_wide(ptr_vector_W[iter_count], ptr_vector_W_grad[iter_count], m_.data(), v_.data(), lr_, w_wide_len, WIDE_LEN);
        auto cpu_end_adam = std::chrono::high_resolution_clock::now();

        auto cpu_begin_sgd= std::chrono::high_resolution_clock::now();
        doWU_SGD_wide(ptr_vector_W[iter_count], ptr_vector_W_grad[iter_count], lr_, w_wide_len, WIDE_LEN);
        auto cpu_end_sgd = std::chrono::high_resolution_clock::now();

        auto cpu_begin_adam_bu= std::chrono::high_resolution_clock::now();
        doWU_ADAM(ptr_vector_B[iter_count], ptr_vector_B_grad[iter_count], b_m_.data(), b_v_.data(), lr_, b_size);
        auto cpu_end_adam_bu = std::chrono::high_resolution_clock::now();

        auto cpu_begin_sgd_bu = std::chrono::high_resolution_clock::now();
        doWU_SGD(ptr_vector_B[iter_count], ptr_vector_B_grad[iter_count], lr_, b_size);
        auto cpu_end_sgd_bu = std::chrono::high_resolution_clock::now();

        auto cpu_dB_begin = std::chrono::high_resolution_clock::now();
        do_dB_conv(ptr_vector_OFM_grad[iter_count], ptr_vector_B_grad[iter_count], rows_in, rows_in, cols_out, lr_, WIDE_LEN);
        auto cpu_dB_end = std::chrono::high_resolution_clock::now();

        std::thread threads[4];

        // SGD multi-threading
        auto cpu_begin_sgd_thread = std::chrono::high_resolution_clock::now();
        for(unsigned i = 0; i < 4; ++i)
        {
            pthread_create(&(sgd_threads[i]), NULL, worker_SGD_wide, (void *)sgd_data_vector[i]);
        }

        for(unsigned i = 0; i < 4; ++i)
        {
            pthread_join(sgd_threads[i], NULL);
        }
        auto cpu_end_sgd_thread = std::chrono::high_resolution_clock::now();

        // ADAM multi-threading
        auto cpu_begin_adam_thread = std::chrono::high_resolution_clock::now();
        for(unsigned i = 0; i < 4; ++i)
        {
            pthread_create(&(adam_threads[i]), NULL, worker_ADAM_wide, (void *)adam_data_vector[i]);
        }

        for(unsigned i = 0; i < 4; ++i)
        {
            pthread_join(adam_threads[i], NULL);
        }
        auto cpu_end_adam_thread = std::chrono::high_resolution_clock::now();

        // dB multi-threading
        auto cpu_begin_db_thread = std::chrono::high_resolution_clock::now();
        for(unsigned i = 0; i < 4; ++i)
        {
            pthread_create(&(bias_threads[i]), NULL, worker_do_dB_conv, (void *)bias_data_vector[i]);
        }

        for(unsigned i = 0; i < 4; ++i)
        {
            pthread_join(bias_threads[i], NULL);
        }
        auto cpu_end_db_thread = std::chrono::high_resolution_clock::now();

        sleep_for(seconds(1));

        //CHECKED THESE
        std::chrono::duration<double> fpga_duration = end_FPGA - begin_FPGA;
        std::chrono::duration<double> fpga_fw = fpga_end_fw - fpga_begin_fw;
        std::chrono::duration<double> fpga_dx = fpga_end_dx - fpga_begin_dx;
        std::chrono::duration<double> fpga_dw = fpga_end_dw - fpga_begin_dw;
        std::chrono::duration<double> wu_sgd = cpu_end_sgd - cpu_begin_sgd;
        std::chrono::duration<double> wu_sgd_thread = cpu_end_sgd_thread - cpu_begin_sgd_thread;
        std::chrono::duration<double> wu_adam = cpu_end_adam - cpu_begin_adam;
        std::chrono::duration<double> wu_adam_thread = cpu_end_adam_thread - cpu_begin_adam_thread;
        std::chrono::duration<double> wu_sgd_bu = cpu_end_sgd_bu - cpu_begin_sgd_bu;
        std::chrono::duration<double> wu_adam_bu = cpu_end_adam_bu - cpu_begin_adam_bu;
        std::chrono::duration<double> cpu_db = cpu_dB_end - cpu_dB_begin;
        std::chrono::duration<double> cpu_db_thread = cpu_end_db_thread - cpu_begin_db_thread;

        printf("- FPGA Time total         : %10.5f ms\n",    fpga_duration.count() * 1000.0);
        printf("- FPGA Time layer FW         : %10.5f ms\n",    fpga_fw.count() * 1000.0);
        printf("- FPGA Time layer dX         : %10.5f ms\n",    fpga_dx.count() * 1000.0);
        printf("- FPGA Time layer dW         : %10.5f ms\n",    fpga_dw.count() * 1000.0);
        printf("- Weight update SGD         : %10.5f ms\n",    wu_sgd.count() * 1000.0);
        printf("- Weight update SGD on multithread  : %10.5f ms\n",    wu_sgd_thread.count() * 1000.0);
        printf("- Weight update ADAM         : %10.5f ms\n",    wu_adam.count() * 1000.0);
        printf("- Weight update ADAM on multithread        : %10.5f ms\n",    wu_adam_thread.count() * 1000.0);
        printf("- Bias update SGD         : %10.5f ms\n",    wu_sgd_bu.count() * 1000.0);
        printf("- Bias update ADAM         : %10.5f ms\n",    wu_adam_bu.count() * 1000.0);
        printf("- CPU time dB         : %10.5f ms\n",    cpu_db.count() * 1000.0);
        printf("- CPU time dB thread         : %10.5f ms\n",    cpu_db_thread.count() * 1000.0);

        float cpu_fw_time = 0.0;
        float cpu_dx_time = 0.0;
        float cpu_dw_time = 0.0;
        float mp = 0.0;
        float mp_bp = 0.0;
        run_conv_on_CPU(cfg, cpu_fw_time, cpu_dx_time, cpu_dw_time, mp, mp_bp);

        OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_vector_IFM[iter_count], ptr_vector_IFM[iter_count]));
        OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_vector_IFM_grad[iter_count], ptr_vector_IFM_grad[iter_count]));
        OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_vector_W[iter_count], ptr_vector_W[iter_count]));
        OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_vector_W_flip[iter_count], ptr_vector_W_flip[iter_count]));
        OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_vector_W_grad[iter_count], ptr_vector_W_grad[iter_count]));
        OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_vector_B[iter_count], ptr_vector_B_grad[iter_count]));
        OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_vector_B_grad[iter_count], ptr_vector_B[iter_count]));
        OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_vector_OFM[iter_count], ptr_vector_OFM[iter_count]));
        OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_vector_OFM_grad[iter_count], ptr_vector_OFM_grad[iter_count]));
        OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_vector_ix[iter_count], ptr_vector_ix[iter_count]));
        OCL_CHECK(err, err = q.finish());

        iter_count++;
    }

    std::cout << "Finished test." << std::endl;

    return 0;
}