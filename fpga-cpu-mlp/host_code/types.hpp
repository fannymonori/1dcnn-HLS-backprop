#ifndef HLSNN_TYPES_H
#define HLSNN_TYPES_H

#include "ap_fixed.h"
#include "hls_vector.h"

#define HLSNN_PRECISION 24
#define HLSNN_INTPART 10

typedef ap_fixed<HLSNN_PRECISION, HLSNN_INTPART> HLSNN_DataType;

#define WIDE_LEN 8

typedef  hls::vector<HLSNN_DataType, WIDE_LEN> wide_type;

#define DEBUG 1

#endif //HLSNN_TYPES_H
