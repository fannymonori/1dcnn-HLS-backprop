#ifndef HLSNN_TYPES_H
#define HLSNN_TYPES_H

#include "ap_fixed.h"

// This is where the precision of the network data type should be adjusted.
#define HLSNN_PRECISION 24
#define HLSNN_INTPART 10

typedef ap_fixed<HLSNN_PRECISION, HLSNN_INTPART> HLSNN_DataType;

#define DEBUG 1

#endif //HLSNN_TYPES_H
