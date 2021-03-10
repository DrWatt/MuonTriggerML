#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 27
#define N_LAYER_2 60
#define N_LAYER_5 50
#define N_LAYER_8 30
#define N_LAYER_11 40
#define N_LAYER_14 15
#define N_LAYER_17 1

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> layer2_t;
typedef ap_fixed<17,2> weight2_t;
typedef ap_fixed<17,2> bias2_t;
typedef ap_fixed<16,6> layer20_t;
typedef ap_fixed<17,2> bias20_t;
typedef ap_fixed<17,2> layer4_t;
typedef ap_fixed<16,6> layer5_t;
typedef ap_fixed<17,2> weight5_t;
typedef ap_fixed<17,2> bias5_t;
typedef ap_fixed<16,6> layer21_t;
typedef ap_fixed<17,2> bias21_t;
typedef ap_fixed<17,2> layer7_t;
typedef ap_fixed<16,6> layer8_t;
typedef ap_fixed<17,2> weight8_t;
typedef ap_fixed<17,2> bias8_t;
typedef ap_fixed<16,6> layer22_t;
typedef ap_fixed<17,2> bias22_t;
typedef ap_fixed<17,2> layer10_t;
typedef ap_fixed<16,6> layer11_t;
typedef ap_fixed<17,2> weight11_t;
typedef ap_fixed<17,2> bias11_t;
typedef ap_fixed<16,6> layer23_t;
typedef ap_fixed<17,2> bias23_t;
typedef ap_fixed<17,2> layer13_t;
typedef ap_fixed<16,6> layer14_t;
typedef ap_fixed<17,2> weight14_t;
typedef ap_fixed<17,2> bias14_t;
typedef ap_fixed<16,6> layer24_t;
typedef ap_fixed<17,2> bias24_t;
typedef ap_fixed<17,2> layer16_t;
typedef ap_fixed<16,6> layer17_t;
typedef ap_fixed<17,2> weight17_t;
typedef ap_fixed<16,6> layer25_t;
typedef ap_fixed<17,2> result_t;

#endif
