#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_large.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/s20.h"
#include "weights/b20.h"
#include "weights/w5.h"
#include "weights/b5.h"
#include "weights/s21.h"
#include "weights/b21.h"
#include "weights/w8.h"
#include "weights/b8.h"
#include "weights/s22.h"
#include "weights/b22.h"
#include "weights/w11.h"
#include "weights/b11.h"
#include "weights/s23.h"
#include "weights/b23.h"
#include "weights/w14.h"
#include "weights/b14.h"
#include "weights/s24.h"
#include "weights/b24.h"
#include "weights/w17.h"
#include "weights/b17.h"
#include "weights/s25.h"
#include "weights/b25.h"

//hls-fpga-machine-learning insert layer-config
struct config2 : nnet::dense_config {
    static const unsigned n_in = N_INPUT_1_1;
    static const unsigned n_out = N_LAYER_2;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 5;
    static const unsigned n_zeros = 1215;
    static const unsigned n_nonzeros = 405;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef bias2_t bias_t;
    typedef weight2_t weight_t;
    typedef ap_uint<1> index_t;
};

struct config20 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_2;
    static const unsigned n_filt = -1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef bias20_t bias_t;
    typedef model_default_t scale_t;
};

struct relu_config4 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 5;
    typedef ap_fixed<18,8> table_t;
};

struct config5 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_2;
    static const unsigned n_out = N_LAYER_5;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 5;
    static const unsigned n_zeros = 2250;
    static const unsigned n_nonzeros = 750;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef bias5_t bias_t;
    typedef weight5_t weight_t;
    typedef ap_uint<1> index_t;
};

struct config21 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_5;
    static const unsigned n_filt = -1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef bias21_t bias_t;
    typedef model_default_t scale_t;
};

struct relu_config7 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_5;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 5;
    typedef ap_fixed<18,8> table_t;
};

struct config8 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_5;
    static const unsigned n_out = N_LAYER_8;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 5;
    static const unsigned n_zeros = 1125;
    static const unsigned n_nonzeros = 375;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef bias8_t bias_t;
    typedef weight8_t weight_t;
    typedef ap_uint<1> index_t;
};

struct config22 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_8;
    static const unsigned n_filt = -1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef bias22_t bias_t;
    typedef model_default_t scale_t;
};

struct relu_config10 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_8;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 5;
    typedef ap_fixed<18,8> table_t;
};

struct config11 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_8;
    static const unsigned n_out = N_LAYER_11;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 5;
    static const unsigned n_zeros = 900;
    static const unsigned n_nonzeros = 300;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef bias11_t bias_t;
    typedef weight11_t weight_t;
    typedef ap_uint<1> index_t;
};

struct config23 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_11;
    static const unsigned n_filt = -1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef bias23_t bias_t;
    typedef model_default_t scale_t;
};

struct relu_config13 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_11;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 5;
    typedef ap_fixed<18,8> table_t;
};

struct config14 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_11;
    static const unsigned n_out = N_LAYER_14;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 5;
    static const unsigned n_zeros = 450;
    static const unsigned n_nonzeros = 150;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef bias14_t bias_t;
    typedef weight14_t weight_t;
    typedef ap_uint<1> index_t;
};

struct config24 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_14;
    static const unsigned n_filt = -1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef bias24_t bias_t;
    typedef model_default_t scale_t;
};

struct relu_config16 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_14;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 5;
    typedef ap_fixed<18,8> table_t;
};

struct config17 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_14;
    static const unsigned n_out = N_LAYER_17;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 5;
    static const unsigned n_zeros = 11;
    static const unsigned n_nonzeros = 4;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef model_default_t bias_t;
    typedef weight17_t weight_t;
    typedef ap_uint<1> index_t;
};

struct config25 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_17;
    static const unsigned n_filt = -1;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef model_default_t bias_t;
    typedef model_default_t scale_t;
};

struct relu_config19 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_17;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 5;
    typedef ap_fixed<18,8> table_t;
};


#endif
