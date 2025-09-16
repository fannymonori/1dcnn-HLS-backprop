#ifndef HLSNN_TYPES_H
#define HLSNN_TYPES_H

#include "ap_fixed.h"
#include "hls_vector.h"

#define HLSNN_PRECISION 32
#define HLSNN_INTPART 16

//typedef ap_fixed<HLSNN_PRECISION, HLSNN_INTPART, AP_RND, AP_SAT> HLSNN_DataType;
typedef ap_fixed<HLSNN_PRECISION, HLSNN_INTPART> HLSNN_DataType;
//typedef half HLSNN_DataType;
//typedef float HLSNN_DataType;

#define WIDE_LEN 8
//#define WIDE_LEN 16

typedef  hls::vector<HLSNN_DataType, WIDE_LEN> wide_type;

typedef ap_uint<16>  index_type;

#define DEBUG 1

#endif //HLSNN_TYPES_H
