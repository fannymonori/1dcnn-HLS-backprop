#ifndef HLSNN_TYPES_H
#define HLSNN_TYPES_H

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_vector.h"

//Change here the data precision if needed
#define HLSNN_PRECISION 24
#define HLSNN_INTPART 10

//Change here the data type if needed
//typedef ap_fixed<HLSNN_PRECISION, HLSNN_INTPART, AP_RND, AP_SAT> HLSNN_DataType;
typedef ap_fixed<HLSNN_PRECISION, HLSNN_INTPART> HLSNN_DataType;
//typedef half HLSNN_DataType;
//typedef float HLSNN_DataType;

typedef float math_type;

//#define WIDE_LEN 16
#define WIDE_LEN 8

typedef  hls::vector<HLSNN_DataType, WIDE_LEN> wide_type;

typedef ap_uint<16> index_type;

#define DEBUG 1

#endif //HLSNN_TYPES_H
