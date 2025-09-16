// ====================================================================================================================
// STRUCT DEFINITIONS

enum LayerType {
    CONV,
    FC
};

class LayerConfig{

    public:

    enum LayerType layer_type;
    unsigned signal_length;
    unsigned input_features;
    unsigned output_features;
    unsigned kernel_size;
    unsigned batch_size;
    
    unsigned rows_in;
    unsigned cols_out;
    unsigned cols_in;
    

    LayerConfig(){
    }

    LayerConfig(LayerType lt, unsigned b, unsigned s, unsigned c, unsigned f, unsigned k){
        layer_type = lt;
        batch_size = b;
        signal_length = s;
        input_features = c;
        output_features = f;
        kernel_size = k;
        if(lt == LayerType::CONV){
            rows_in = output_features;
            cols_out = signal_length;
            cols_in = kernel_size * input_features;
        }
        else{
            rows_in = b;
            cols_out = output_features;
            cols_in = input_features;
        }

    }

    //friend ostream& operator<<(ostream& cout, LayerConfig& l_cfg);

    void print(std::ofstream &out_file) 
    { 
        out_file << "Layer Config: " << std::endl;

        if(layer_type == LayerType::CONV){
            out_file << "layer type: CONV" << std::endl;
            out_file << "signal length: " << std::to_string(signal_length) << std::endl;
            out_file << "input features: " << std::to_string(input_features) << std::endl;
            out_file << "output features: " << std::to_string(output_features) << std::endl;
            out_file << "kernel size: " << std::to_string(kernel_size) << std::endl;
            out_file << "im2col rows in: " << std::to_string(rows_in) << std::endl;
            out_file << "im2col cols out: " << std::to_string(cols_out) << std::endl;
            out_file << "im2col cols in: " << std::to_string(cols_in) << std::endl;
        }
        else{
            out_file << "layer type: FC" << std::endl;
            out_file << "batch size: " << std::to_string(batch_size) << std::endl;
            out_file << "input neurons: " << std::to_string(input_features) << std::endl;
            out_file << "output neurons: " << std::to_string(output_features) << std::endl;
            out_file << "mm rows in: " << std::to_string(rows_in) << std::endl;
            out_file << "mm cols out: " << std::to_string(cols_out) << std::endl;
            out_file << "mm cols in: " << std::to_string(cols_in) << std::endl;
        }
    } 

};