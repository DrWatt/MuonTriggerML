#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tensorflow.random import set_seed
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Activation
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.callbacks import History 
import matplotlib.pyplot as plt
from keras.models import load_model
from subprocess import check_output
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu


from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning

seed = 1563
np.random.seed(seed)
set_seed(seed)

outrootname = "QKeras_Model_4_Pruned_120"
Nfolder = check_output("if [ ! -d \"" + outrootname + "\" ]; then mkdir \"" + outrootname + "\";fi",shell=True)
Nfolder = check_output("if [ ! -d \"" + outrootname + "/hist\" ]; then mkdir \"" + outrootname + "/hist\";fi",shell=True)


def preprocess_features(muon_dataframe):
    """Prepares input features from Muon data set.

    Args:
        muon_dataframe: A Pandas DataFrame expected to contain data
        from muon simulations
    Returns:
        A DataFrame that contains the features to be used for the model.
  """
    selected_features = muon_dataframe[
      [#'Event',
    'n_Primitive',
    '1dtPrimitive.id_r',
    '2dtPrimitive.id_r',
    '3dtPrimitive.id_r',
    '4dtPrimitive.id_r',
    '1dtPrimitive.id_eta',
    '2dtPrimitive.id_eta',
    '3dtPrimitive.id_eta',
    '4dtPrimitive.id_eta',
    '1dtPrimitive.id_phi',
    '2dtPrimitive.id_phi',
    '3dtPrimitive.id_phi',
    '4dtPrimitive.id_phi',
    '1dtPrimitive.phiB',
    '2dtPrimitive.phiB',
    '3dtPrimitive.phiB',
    '4dtPrimitive.phiB',
    '1dtPrimitive.quality',
    '2dtPrimitive.quality',
    '3dtPrimitive.quality',
    '4dtPrimitive.quality',
    'delta_phi12',
    'delta_phi13',
    'delta_phi14',
    'delta_phi23',
    'delta_phi24',
    'delta_phi34'
         ]]
    processed_features = selected_features.copy()
    return processed_features.astype(np.float64)

def preprocess_targets(muon_dataframe):
    """
    Prepares target features (i.e., labels) from muon data set.

    Args:
        muon_dataframe: A Pandas DataFrame expected to contain data
        from the Muon data set.
    Returns:
        A DataFrame that contains the target feature.
    """
    output_targets = pd.DataFrame()
    max_value = muon_dataframe["genParticle.pt"].max()
    min_value = muon_dataframe["genParticle.pt"].min()
    #output_targets["genParticle.pt"] = (muon_dataframe["genParticle.pt"]-min_value)/(max_value-min_value)
    output_targets["genParticle.pt"] = muon_dataframe["genParticle.pt"]/200.
    return output_targets.astype(np.float32)


def datasets():
    out_dataframe = pd.read_csv('../output/bxcut_full_muon.csv')
    muon_dataframe_test = pd.read_csv('../output/bxcut_full_test.csv')
    
    out_dataframe["1dtPrimitive.phiB"] = out_dataframe["1dtPrimitive.phiB"]/512.
    out_dataframe["2dtPrimitive.phiB"] = out_dataframe["2dtPrimitive.phiB"]/512.
    out_dataframe["3dtPrimitive.phiB"] = out_dataframe["3dtPrimitive.phiB"]/512.
    out_dataframe["4dtPrimitive.phiB"] = out_dataframe["4dtPrimitive.phiB"]/512.
    
    muon_dataframe_test["1dtPrimitive.phiB"] = muon_dataframe_test["1dtPrimitive.phiB"]/512.
    muon_dataframe_test["2dtPrimitive.phiB"] = muon_dataframe_test["2dtPrimitive.phiB"]/512.
    muon_dataframe_test["3dtPrimitive.phiB"] = muon_dataframe_test["3dtPrimitive.phiB"]/512.
    muon_dataframe_test["4dtPrimitive.phiB"] = muon_dataframe_test["4dtPrimitive.phiB"]/512.
    
    out_dataframe = out_dataframe[out_dataframe['genParticle.pt'] <= 200]
    
    #Creating different datasets
    
    out_dataframe60 = out_dataframe[out_dataframe['genParticle.pt'] <= 60]
    out_dataframe120 = out_dataframe[out_dataframe['genParticle.pt'] <= 120]
    out_dataframe120 = out_dataframe120[out_dataframe120['genParticle.pt'] > 60]
    out_dataframe180 = out_dataframe[out_dataframe['genParticle.pt'] <= 180]
    out_dataframe180 = out_dataframe180[out_dataframe180['genParticle.pt'] > 120]
    

    
    
    
    out_dataframe60.to_csv("out60.csv")
    out_dataframe120.to_csv("out120.csv")
    out_dataframe180.to_csv("out180.csv")
    
    
    out_dataframe.to_csv("out_data.csv")
    
    muon_dataframe_test = muon_dataframe_test[muon_dataframe_test['genParticle.pt'] <= 200]
    
    muon_dataframe_test60 = muon_dataframe_test[muon_dataframe_test['genParticle.pt'] <= 60]
    muon_dataframe_test120 = muon_dataframe_test[muon_dataframe_test['genParticle.pt'] <= 120]
    muon_dataframe_test120 = muon_dataframe_test120[muon_dataframe_test120['genParticle.pt'] > 60]
    muon_dataframe_test180 = muon_dataframe_test[muon_dataframe_test['genParticle.pt'] <= 180]
    muon_dataframe_test180 = muon_dataframe_test180[muon_dataframe_test180['genParticle.pt'] > 120]
    
    muon_dataframe_test60.to_csv("test60.csv")
    muon_dataframe_test120.to_csv("test120.csv")
    muon_dataframe_test180.to_csv("test180.csv")
    
    
    muon_dataframe_test.to_csv("muon_test.csv")

    return 0

def preproc(test=False):
    
    out_dataframe = pd.read_csv('out120.csv')
    muon_dataframe_test = pd.read_csv('test120.csv')
    
    X = preprocess_features(out_dataframe)
    X_test = preprocess_features(muon_dataframe_test)
    
    Y = preprocess_targets(out_dataframe)
    Y_test = preprocess_targets(muon_dataframe_test)
    Y_test = Y_test.fillna(0)
    
    X.loc[X["1dtPrimitive.quality"] < 4, '1dtPrimitive.quality'] = 0.0
    X.loc[X["1dtPrimitive.quality"] >= 4, '1dtPrimitive.quality'] = 1.0
    X.loc[X["2dtPrimitive.quality"] < 4, '2dtPrimitive.quality'] = 0.0
    X.loc[X["2dtPrimitive.quality"] >= 4, '2dtPrimitive.quality'] = 1.0
    X.loc[X["3dtPrimitive.quality"] < 4, '3dtPrimitive.quality'] = 0.0
    X.loc[X["3dtPrimitive.quality"] >= 4, '3dtPrimitive.quality'] = 1.0
    X.loc[X["4dtPrimitive.quality"] < 4, '4dtPrimitive.quality'] = 0.0
    X.loc[X["4dtPrimitive.quality"] >= 4, '4dtPrimitive.quality'] = 1.0
    
    X_test.loc[X_test["1dtPrimitive.quality"] < 4, '1dtPrimitive.quality'] = 0.0
    X_test.loc[X_test["1dtPrimitive.quality"] >= 4, '1dtPrimitive.quality'] = 1.0
    X_test.loc[X_test["2dtPrimitive.quality"] < 4, '2dtPrimitive.quality'] = 0.0
    X_test.loc[X_test["2dtPrimitive.quality"] >= 4, '2dtPrimitive.quality'] = 1.0
    X_test.loc[X_test["3dtPrimitive.quality"] < 4, '3dtPrimitive.quality'] = 0.0
    X_test.loc[X_test["3dtPrimitive.quality"] >= 4, '3dtPrimitive.quality'] = 1.0
    X_test.loc[X_test["4dtPrimitive.quality"] < 4, '4dtPrimitive.quality'] = 0.0
    X_test.loc[X_test["4dtPrimitive.quality"] >= 4, '4dtPrimitive.quality'] = 1.0


    
    
    if test == True:
        return X_test,Y_test
    else:
        return X,Y





def Q_baseline_model(size,epochs,optimizer,X_training,y_training,X_validation,y_validation,output_name):
    
    pruning = True
    # create model
    name="RMSE validation"
    name2="RMSE training"
    history = History()
    model = Sequential()
    model.add(QDense(50,  input_shape=(27,),kernel_quantizer=quantized_bits(16,0),bias_quantizer=quantized_bits(16,0), kernel_initializer='random_normal'))
    model.add(QActivation(activation=quantized_relu(16), name='relu1'))
    # model.add(Dropout(rate=0.2))
    model.add(QDense(25,kernel_quantizer=quantized_bits(16,0),bias_quantizer=quantized_bits(16,0)))
    model.add(QActivation(activation=quantized_relu(16), name='relu2'))
    model.add(QDense(10,kernel_quantizer=quantized_bits(16,0),bias_quantizer=quantized_bits(16,0)))
    model.add(QActivation(activation=quantized_relu(16), name='relu3'))
    # model.add(Dropout(rate=0.2))
    model.add(Dense(1,activation='sigmoid'))
    #model.add(QActivation(activation=hard_sigmoid(6), name='sigm3'))
    #model.add(Activation("sigmoid"))
    
    if pruning == True:
        print("////////////////////////Training Model with pruning")
        pruning_params = {"pruning_schedule" : pruning_schedule.ConstantSparsity(0.75, begin_step=2000, frequency=100)}
        model = prune.prune_low_magnitude(model, **pruning_params)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        model.fit(X_training, y_training,
                  batch_size=size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(X_validation, y_validation),
                  callbacks=[history,pruning_callbacks.UpdatePruningStep()])
        
        model = strip_pruning(model)
        w = model.layers[0].weights[0].numpy()
        h, b = np.histogram(w, bins=100)
        plt.figure(figsize=(7,7))
        plt.bar(b[:-1], h, width=b[1]-b[0])
        plt.semilogy()
        plt.savefig("Zeros' distribution",format='png')
        print('% of zeros = {}'.format(np.sum(w==0)/np.size(w)))
    else:
        print("////////////////////////Training Model WITHOUT pruning")
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        model.fit(X_training, y_training,
                  batch_size=size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(X_validation, y_validation),
                  callbacks=[history])
    # Compile model
    # model.compile(loss='mean_squared_error', optimizer=optimizer)
    # model.fit(X_training, y_training,
    #       batch_size=size,
    #       epochs=epochs,
    #       verbose=1,
    #       validation_data=(X_validation, y_validation),callbacks=[history])
    
    w = []
    for layer in model.layers:
        print(layer)
        w.append(layer.get_weights())
    
    #print(w)
    train_predictions = model.predict(X_training)
    predictions = model.predict(X_validation)
    lin_mse = mean_squared_error(y_validation, predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_mse2 = mean_squared_error(y_training, train_predictions)
    lin_rmse2 = np.sqrt(lin_mse2)
    msg = "%s: %f" % (name, lin_rmse)
    msg2 = "%s: %f" % (name2, lin_rmse2)
    print(msg)
    print(msg2)
    fig,ax = plt.subplots()
    # xy=np.vstack([y_validation, predictions])
    #z=gaussian_kde(xy)
    ax.scatter(y_validation, predictions, edgecolors=(0, 0, 0))
    ax.set_title('Regression model predictions (validation set)')
    ax.set_xlabel('Measured $p_T$ (GeV/c)')
    ax.set_ylabel('Predicted $p_T$ (GeV/c)')
    ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
    plt.rc('font', size=20)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)    
    plt.rc('xtick', labelsize=18)   
    plt.rc('ytick', labelsize=18)  
    plt.rc('legend', fontsize=18)    
    plt.rc('figure', titlesize=18)
    plt.tight_layout()
    plt.savefig(outrootname + '/' +'1'+ output_name,format='png',dpi=800)
    fig2,ax2 = plt.subplots()
    ax2.plot(history.history['loss'], label='loss')
    ax2.plot(history.history['val_loss'], label='val_loss')
    ax2.set_title('Training and Validation loss per epoch')
    ax2.set_xlabel('# Epoch')
    ax2.set_ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outrootname + '/' +'2'+ output_name,format='png',dpi=800)
    #plt.show()
    del ax,ax2
    

    return model,w




















X,Y = preproc()



def run(X,Y):
    x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2,random_state=seed)
    adam = Adam(lr=0.0001) #Default 0.001
    model,w = Q_baseline_model(100,50,adam,x_train, y_train, x_valid, y_valid,'Adam.png')
    model.save(outrootname + '/' + outrootname + '.h5')
    np.savetxt(outrootname + '/' +"hist/weights_1.csv",w[0][0],delimiter=',')
    np.savetxt(outrootname + '/' +"hist/weights_2.csv",w[2][0],delimiter=',')
    np.savetxt(outrootname + '/' +"hist/weights_3.csv",w[4][0],delimiter=',')
    np.savetxt(outrootname + '/' +"hist/weights_3.csv",w[6][0],delimiter=',')
    np.savetxt(outrootname + '/' +"hist/bias_1.csv",w[0][1],delimiter=',')
    np.savetxt(outrootname + '/' +"hist/bias_2.csv",w[2][1],delimiter=',')
    np.savetxt(outrootname + '/' +"hist/bias_3.csv",w[4][1],delimiter=',')
    np.savetxt(outrootname + '/' +"hist/bias_3.csv",w[6][1],delimiter=',')
    return model,w

