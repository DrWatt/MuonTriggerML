//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>
#include "/mnt/sda4/tools/Xilinx/Vivado/2020.1/include/gmp.h"

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t q_dense_10_input[N_INPUT_1_1],
    result_t layer19_out[N_LAYER_17],
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1
) {

    //hls-fpga-machine-learning insert IO
	#pragma HLS ARRAY_RESHAPE variable=q_dense_10_input complete dim=0
	#pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
	#pragma HLS INTERFACE s_axilite port=q_dense_10_input bundle=input
	#pragma HLS INTERFACE s_axilite port=layer19_out bundle=output
	#pragma HLS INTERFACE s_axilite port=return bundle=input

    //#pragma HLS ARRAY_RESHAPE variable=q_dense_10_input complete dim=0
    //#pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
    //#pragma HLS INTERFACE ap_vld port=q_dense_10_input,layer19_out
    #pragma HLS PIPELINE II=5

    const_size_in_1 = N_INPUT_1_1;
    const_size_out_1 = N_LAYER_17;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight2_t, 1620>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 60>(b2, "b2.txt");
        nnet::load_weights_from_txt<model_default_t, 60>(s20, "s20.txt");
        nnet::load_weights_from_txt<bias20_t, 60>(b20, "b20.txt");
        nnet::load_weights_from_txt<weight5_t, 3000>(w5, "w5.txt");
        nnet::load_weights_from_txt<bias5_t, 50>(b5, "b5.txt");
        nnet::load_weights_from_txt<model_default_t, 50>(s21, "s21.txt");
        nnet::load_weights_from_txt<bias21_t, 50>(b21, "b21.txt");
        nnet::load_weights_from_txt<weight8_t, 1500>(w8, "w8.txt");
        nnet::load_weights_from_txt<bias8_t, 30>(b8, "b8.txt");
        nnet::load_weights_from_txt<model_default_t, 30>(s22, "s22.txt");
        nnet::load_weights_from_txt<bias22_t, 30>(b22, "b22.txt");
        nnet::load_weights_from_txt<weight11_t, 1200>(w11, "w11.txt");
        nnet::load_weights_from_txt<bias11_t, 40>(b11, "b11.txt");
        nnet::load_weights_from_txt<model_default_t, 40>(s23, "s23.txt");
        nnet::load_weights_from_txt<bias23_t, 40>(b23, "b23.txt");
        nnet::load_weights_from_txt<weight14_t, 600>(w14, "w14.txt");
        nnet::load_weights_from_txt<bias14_t, 15>(b14, "b14.txt");
        nnet::load_weights_from_txt<model_default_t, 15>(s24, "s24.txt");
        nnet::load_weights_from_txt<bias24_t, 15>(b24, "b24.txt");
        nnet::load_weights_from_txt<weight17_t, 15>(w17, "w17.txt");
        nnet::load_weights_from_txt<model_default_t, 1>(b17, "b17.txt");
        nnet::load_weights_from_txt<model_default_t, 1>(s25, "s25.txt");
        nnet::load_weights_from_txt<model_default_t, 1>(b25, "b25.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::dense_latency<input_t, layer2_t, config2>(q_dense_10_input, layer2_out, w2, b2);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer2_t>(layer2_out, "q_dense_10", N_LAYER_2);
#endif

    layer20_t layer20_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer20_out complete dim=0
    nnet::normalize<layer2_t, layer20_t, config20>(layer2_out, layer20_out, s20, b20);

    layer4_t layer4_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::relu<layer20_t, layer4_t, relu_config4>(layer20_out, layer4_out);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer4_t>(layer4_out, "relu1", N_LAYER_2);
#endif

    layer5_t layer5_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::dense_latency<layer4_t, layer5_t, config5>(layer4_out, layer5_out, w5, b5);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer5_t>(layer5_out, "q_dense_11", N_LAYER_5);
#endif

    layer21_t layer21_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer21_out complete dim=0
    nnet::normalize<layer5_t, layer21_t, config21>(layer5_out, layer21_out, s21, b21);

    layer7_t layer7_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::relu<layer21_t, layer7_t, relu_config7>(layer21_out, layer7_out);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer7_t>(layer7_out, "relu2", N_LAYER_5);
#endif

    layer8_t layer8_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::dense_latency<layer7_t, layer8_t, config8>(layer7_out, layer8_out, w8, b8);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer8_t>(layer8_out, "q_dense_12", N_LAYER_8);
#endif

    layer22_t layer22_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0
    nnet::normalize<layer8_t, layer22_t, config22>(layer8_out, layer22_out, s22, b22);

    layer10_t layer10_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::relu<layer22_t, layer10_t, relu_config10>(layer22_out, layer10_out);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer10_t>(layer10_out, "relu3", N_LAYER_8);
#endif

    layer11_t layer11_out[N_LAYER_11];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::dense_latency<layer10_t, layer11_t, config11>(layer10_out, layer11_out, w11, b11);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer11_t>(layer11_out, "q_dense_13", N_LAYER_11);
#endif

    layer23_t layer23_out[N_LAYER_11];
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0
    nnet::normalize<layer11_t, layer23_t, config23>(layer11_out, layer23_out, s23, b23);

    layer13_t layer13_out[N_LAYER_11];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::relu<layer23_t, layer13_t, relu_config13>(layer23_out, layer13_out);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer13_t>(layer13_out, "relu4", N_LAYER_11);
#endif

    layer14_t layer14_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::dense_latency<layer13_t, layer14_t, config14>(layer13_out, layer14_out, w14, b14);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer14_t>(layer14_out, "q_dense_14", N_LAYER_14);
#endif

    layer24_t layer24_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer24_out complete dim=0
    nnet::normalize<layer14_t, layer24_t, config24>(layer14_out, layer24_out, s24, b24);

    layer16_t layer16_out[N_LAYER_14];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    nnet::relu<layer24_t, layer16_t, relu_config16>(layer24_out, layer16_out);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer16_t>(layer16_out, "relu5", N_LAYER_14);
#endif

    layer17_t layer17_out[N_LAYER_17];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    nnet::dense_latency<layer16_t, layer17_t, config17>(layer16_out, layer17_out, w17, b17);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer17_t>(layer17_out, "q_dense_15", N_LAYER_17);
#endif

    layer25_t layer25_out[N_LAYER_17];
    #pragma HLS ARRAY_PARTITION variable=layer25_out complete dim=0
    nnet::normalize<layer17_t, layer25_t, config25>(layer17_out, layer25_out, s25, b25);

    nnet::relu<layer25_t, result_t, relu_config19>(layer25_out, layer19_out);
#ifndef __SYNTHESIS__
    nnet::save_layer_output<result_t>(layer19_out, "relu6", N_LAYER_17);
#endif

}
