#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from keras.models import load_model
from qkeras.utils import load_qmodel
from QKeras_Model import preproc
import hls4ml
from subprocess import check_output

modelname = "QKeras_Model_60_50_30_40_15"
outrootname ="QKeras_Model_60_50_30_40_15_HLS_noreuse"

#Preprocessing data
X_test,Y_test = preproc(True)

#Loading QKeras model from disk
model = load_qmodel(modelname + '/' + modelname + '.h5')

#Creating configuration dictionary
config = hls4ml.utils.config_from_keras_model(model, granularity='name',default_reuse_factor=1)

#Activating tracing (i.e. saving also the results passed between hidden layers)
for layer in config['LayerName'].keys():
    config['LayerName'][layer]['Trace'] = True

#Creating HLS_model from the QKeras one using the config dict
hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=modelname +'/'+ outrootname + '/HLS_Project',fpga_part='xczu9eg-ffvb1156-2-e')

#Building project directory with .tcl script for Vivado_hls containing instruction for the synthesis.
hls_model.compile()

#Saving tracing output in csv files
hls4ml_pred, hls4ml_trace = hls_model.trace(np.ascontiguousarray(X_test.to_numpy()))
keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, np.ascontiguousarray(X_test.to_numpy()))
np_keras_trace = {}
for key in keras_trace.keys():
    np_keras_trace[key] = keras_trace[key].numpy()
check_output("if [ ! -d \"" + modelname +"/"+ outrootname + "/keras_trace\" ]; then mkdir \"" + modelname +"/"+ outrootname + "/keras_trace\";fi",shell=True)
check_output("if [ ! -d \"" + modelname +"/"+ outrootname + "/hls4ml_trace\" ]; then mkdir \"" + modelname +"/"+ outrootname + "/hls4ml_trace\";fi",shell=True)
for key in np_keras_trace.keys():
    np.savetxt(modelname +"/"+outrootname + "/keras_trace/"+key+".csv",np_keras_trace[key],delimiter=',')
for key in hls4ml_trace.keys():
    np.savetxt(modelname +"/"+outrootname + "/hls4ml_trace/"+key+".csv",hls4ml_trace[key],delimiter=',')
