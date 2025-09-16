const float beta_1=0.9; // decay rate of first moment estimate
const float beta_2=0.999; // decay rate of second moment estimate

const float inv_beta_1=0.1;
const float inv_beta_2=0.001;

const float epsilon=0.000001;

const float learning_rate=0.001;

template<class T>
void doWU_ADAM(T* w_orig,
    T* w,
    float* m_array,
    float* v_array,
    float LR,
    unsigned length){

    float m_t, v_t, m_t_, v_t_;
    float w_prev, w_grad;

    float lr = float(LR);

    for(unsigned f = 0; f < length; f++){
        
        w_prev = float(w_orig[f]);
        w_grad = float(w[f]);
        m_t = beta_1 * m_array[f] + inv_beta_1 * w_grad;
        v_t = beta_2 * v_array[f] + inv_beta_2 * (w_grad * w_grad);

    	m_t_ = m_t / (1 - beta_1);
    	v_t_ = v_t / (1 - beta_2);

    	w_orig[f] = T(w_prev - lr * m_t_ / (std::sqrt(std::abs(v_t_)) + epsilon));
    }

}

void doWU_ADAM_wide(wide_type* w_orig,
    wide_type* w,
    float* m_array,
    float* v_array,
    float LR,
    unsigned length,
    unsigned wide_length){

    float m_t, v_t, m_t_, v_t_;
    float w_prev, w_grad;

    float lr = float(LR);

    for(unsigned f = 0; f < length; f++){
        for(unsigned j = 0; j < wide_length; j++){
            w_prev = float(w_orig[f][j]);
            w_grad = float(w[f][j]);
            m_t = beta_1 * m_array[f] + inv_beta_1 * w_grad;
            v_t = beta_2 * v_array[f] + inv_beta_2 * (w_grad * w_grad);

            m_t_ = m_t / (1 - beta_1);
            v_t_ = v_t / (1 - beta_2);

            w_orig[f][j] = HLSNN_DataType(w_prev - lr * m_t_ / (std::sqrt(std::abs(v_t_)) + epsilon));
        }
    }

}


template<class T>
void doWU_SGD(T* w_orig, T* w, float LR, unsigned length){

    for(unsigned f = 0; f < length; f++){
        w_orig[f] = T(float(w_orig[f]) - LR * float(w[f]));
    }

}

void doWU_SGD_wide(wide_type* w_orig, wide_type* w, float LR, unsigned length, unsigned wide_length){

    for(unsigned f = 0; f < length; f++){
        for(unsigned j = 0; j < wide_length; j++){
            w_orig[f][j] = HLSNN_DataType(float(w_orig[f][j]) - LR * float(w[f][j]));
        }
    }

}


// ====================================================================================================================
// WEIGHT UPDATE MULTI-THREADING

struct SGD_thread_data{
    HLSNN_DataType* w_orig;
    HLSNN_DataType* w;
    float LR;
    unsigned length;
    unsigned start;
    unsigned end;
};

struct SGD_thread_data_wide{
    wide_type* w_orig;
    wide_type* w;
    float LR = 0.001;
    unsigned length = 0;
    unsigned start = 0;
    unsigned end = 0;
    unsigned wide_length = 0;
};

//template<class T>
struct ADAM_thread_data{
    HLSNN_DataType* w_orig;
    HLSNN_DataType* w;
    float* m_array;
    float* v_array;
    float LR;
    unsigned length;
};

struct ADAM_thread_data_wide{
    wide_type* w_orig;
    wide_type* w;
    float* m_array;
    float* v_array;
    float LR;
    unsigned time_step;
    unsigned length;
    unsigned start;
    unsigned end;
    unsigned wide_length;
};


void *worker_SGD(void *args){

    struct SGD_thread_data *args_ = ((struct SGD_thread_data*) args);

    float learning_rate = args_->LR;
    unsigned start = args_->start;
    unsigned end = args_->end;

    for(unsigned f = start; f < end; f++){
        args_->w_orig[f] = float(args_->w_orig[f]) - learning_rate * float(args_->w[f]);
    }

}

/*
This function is for performing one step of SGD parameter update on wide data type
*/
void *worker_SGD_wide(void *args){

    struct SGD_thread_data_wide *args_ = ((struct SGD_thread_data_wide*) args);

    float learning_rate = args_->LR;
    //float learning_rate = 0.001;

    wide_type* w_orig = args_->w_orig;
    wide_type* w = args_->w;
    unsigned start = args_->start;
    unsigned end = args_->end;
    unsigned wide_length = args_->wide_length;

    unsigned f = 0;
    for(f = start; f < end; f++){
        for(unsigned j = 0; j < wide_length; j++){
            w_orig[f][j] = float(w_orig[f][j]) - learning_rate * float(w[f][j]);
        }
    }

}

/*
This function is for performing one step of ADAM parameter update on simple data type
*/
void *worker_ADAM(void *args){

    struct ADAM_thread_data *args_ = ((struct ADAM_thread_data*) args);

    float m_t, v_t, m_t_, v_t_;
    float w_prev, w_grad;

    float lr = float(args_->LR);

    HLSNN_DataType* w_orig = args_->w_orig;
    HLSNN_DataType* w = args_->w;
    float* m_array = args_->m_array;
    float* v_array = args_->v_array;
    unsigned length = args_->length;


    for(unsigned f = 0; f < length; f++){
        
        w_prev = float(w_orig[f]);
        w_grad = float(w[f]);
        m_t = beta_1 * m_array[f] + inv_beta_1 * w_grad;
        v_t = beta_2 * v_array[f] + inv_beta_2 * (w_grad * w_grad);

    	m_t_ = m_t / (1 - beta_1);
    	v_t_ = v_t / (1 - beta_2);

    	w_orig[f] = HLSNN_DataType(w_prev - lr * m_t_ / (std::sqrt(std::abs(v_t_)) + epsilon));
    }
}

/*
This function is for performing one step of ADAM parameter update on wide data type.
*/
void *worker_ADAM_wide(void *args){

    struct ADAM_thread_data_wide *args_ = ((struct ADAM_thread_data_wide*) args);

    float m_t, v_t, m_t_, v_t_;
    float w_prev, w_grad;

    //float lr = float(args_->LR);
    float lr = 0.001;

    wide_type* w_orig = args_->w_orig;
    wide_type* w = args_->w;
    float* m_array = args_->m_array;
    float* v_array = args_->v_array;
    unsigned wide_length = args_->wide_length;
    unsigned start = args_->start;
    unsigned end = args_->end;
    unsigned time_step = args_->time_step;

    float beta1_t = pow(beta_1, time_step);
    float beta2_t = pow(beta_2, time_step);

    unsigned c = 0;
    for(unsigned f = start; f < end; f++){
        for(unsigned j = 0; j < wide_length; j++){
            w_prev = float(w_orig[f][j]);
            w_grad = float(w[f][j]);
            m_t = beta_1 * m_array[c] + inv_beta_1 * w_grad;
            v_t = beta_2 * v_array[c] + inv_beta_2 * (w_grad * w_grad);

            m_array[c] = m_t;
            v_array[c] = v_t;

            m_t_ = m_t / (1 - beta1_t);
            v_t_ = v_t / (1 - beta2_t);

            // Weight update
            w_orig[f][j] = HLSNN_DataType(w_prev - lr * m_t_ / (std::sqrt(std::abs(v_t_)) + epsilon));
            c++;
        }
    }
}

