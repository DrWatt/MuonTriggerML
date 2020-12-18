#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from keras.models import load_model
from qkeras.utils import load_qmodel
from QKeras_Model import preproc
import hls4ml
from subprocess import check_output

outrootname = "QKeras_Model_4"

X_test,Y_test = preproc(True)
# outputFile = root.TFile(outrootname + '/' + outrootname +"_hls.root","recreate")
# c1 = root.TCanvas("c1","c1");
name="RMSE"
model = load_qmodel(outrootname + '/' + outrootname + '.h5')

config = hls4ml.utils.config_from_keras_model(model, granularity='name')
#print(config)

for layer in config['LayerName'].keys():
    config['LayerName'][layer]['Trace'] = True

#                   BIG NET
# config['LayerName']['dense']['Precision']['weight'] = 'ap_fixed<31,9>'
# config['LayerName']['dense']['Precision']['bias'] = 'ap_fixed<10,3>'
# config['LayerName']['dense']['Precision']['result'] = 'ap_fixed<52,17>'
# #config['LayerName']['dense_sigmoid']['Precision'] = 'ap_fixed<32,5>'
# #config['LayerName']['dense_sigmoid']['table_size'] = '4096'
# config['LayerName']['dense_sigmoid']['table_t'] = 'ap_fixed<52,17>'
# config['LayerName']['dense_1']['Precision']['weight'] = 'ap_fixed<28,7>'
# config['LayerName']['dense_1']['Precision']['bias'] = 'ap_fixed<11,4>'
# config['LayerName']['dense_1']['Precision']['result'] = 'ap_fixed<49,18>'
# #config['LayerName']['dense_1_sigmoid']['Precision'] = 'ap_fixed<32,5>'
# #config['LayerName']['dense_1_sigmoid']['table_size'] = '4096'
# config['LayerName']['dense_1_sigmoid']['table_t'] = 'ap_fixed<49,18>'
# config['LayerName']['dense_2']['Precision']['weight'] = 'ap_fixed<11,3>'
# config['LayerName']['dense_2']['Precision']['bias'] = 'ap_fixed<9,2>'
# config['LayerName']['dense_2']['Precision']['result'] = 'ap_fixed<31,13>'
# #config['LayerName']['dense_2_sigmoid']['Precision'] = 'ap_fixed<32,5>'
# #config['LayerName']['dense_2_sigmoid']['table_size'] = '4096'
# config['LayerName']['dense_2_sigmoid']['table_t'] = 'ap_fixed<31,13>'
# config['LayerName']['input1']['Precision'] = 'ap_fixed<22,5>'


#            SMALL NET

# config['LayerName']['dense']['Precision']['weight'] = 'ap_fixed<25,9>'
# config['LayerName']['dense']['Precision']['bias'] = 'ap_fixed<11,4>'
# config['LayerName']['dense']['Precision']['result'] = 'ap_fixed<47,18>'
# #config['LayerName']['dense_sigmoid']['Precision'] = 'ap_fixed<32,5>'
# #config['LayerName']['dense_sigmoid']['table_size'] = '4096'
# config['LayerName']['dense_sigmoid']['table_t'] = 'ap_fixed<47,18>'
# config['LayerName']['dense_1']['Precision']['weight'] = 'ap_fixed<12,4>'
# config['LayerName']['dense_1']['Precision']['bias'] = 'ap_fixed<7,1>'
# config['LayerName']['dense_1']['Precision']['result'] = 'ap_fixed<33,15>'
# #config['LayerName']['dense_1_sigmoid']['Precision'] = 'ap_fixed<32,5>'
# #config['LayerName']['dense_1_sigmoid']['table_size'] = '4096'
# config['LayerName']['dense_1_sigmoid']['table_t'] = 'ap_fixed<33,15>'
# config['LayerName']['input1']['Precision'] = 'ap_fixed<22,5>'



hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=outrootname + '/HLS_Project',fpga_part='xqzu19eg-ffrb1517-2-i')
#hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file="TEST_HLS.png")
hls_model.compile()





# numxtest=X_test.to_numpy()
# print(numxtest[0])
# print('\n')
# print(numxtest[10])

# print('\n')
# print(numxtest[1])    
# print('\n')
# print(numxtest[11])
# predictions = hls_model.predict(np.ascontiguousarray(numxtest))


# np.savetxt("out_hls.csv",predictions,delimiter=',')

#prof = hls4ml.model.profiling.numerical(keras_model=model, hls_model=hls_model)
hls4ml_pred, hls4ml_trace = hls_model.trace(np.ascontiguousarray(X_test.to_numpy()))
keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, np.ascontiguousarray(X_test.to_numpy()))
np_keras_trace = {}
for key in keras_trace.keys():
    np_keras_trace[key] = keras_trace[key].numpy()
check_output("if [ ! -d \"" + outrootname + "/keras_trace\" ]; then mkdir \"" + outrootname + "/keras_trace\";fi",shell=True)
check_output("if [ ! -d \"" + outrootname + "/hls4ml_trace\" ]; then mkdir \"" + outrootname + "/hls4ml_trace\";fi",shell=True)
for key in np_keras_trace.keys():
    np.savetxt(outrootname + "/keras_trace/"+key+".csv",np_keras_trace[key],delimiter=',')
for key in hls4ml_trace.keys():
    np.savetxt(outrootname + "/hls4ml_trace/"+key+".csv",hls4ml_trace[key],delimiter=',')
