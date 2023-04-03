# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:23:22 2023

@author: Josef Tausendschön
"""

import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import model_from_json

"""------------------------define network to load------------------------------"""

opti  = keras.optimizers.Adam         # the optimizer must be specified, however has no influence
loss  = 'mean_absolute_error'
alpha = 0.0005

file0 = 'PP_4Marker_DNN/PP_DNN_4markers_structure.json'
file1 = 'PP_4Marker_DNN/PP_DNN_4markers_weights.h5'

max_ = np.array([0.79447273, 33.0, 2.5, 0.55412932])
min_ = np.array([0.01530295,  0.0, 0.4, 0.19921297])

maxY = 0.2108413
minY = 0.0
""" ########################################################################## """

"""------------------------load and compile dnn------------------------------"""    

json_file         = open(file0,'r')                         # open json file
loaded_model_json = json_file.read()                        # read json file
json_file.close()                                           # close json file

loaded_modelDNN = model_from_json(loaded_model_json)           # load structure of .json file

loaded_modelDNN.load_weights(file1)                            # load weights of .h5 file

loaded_modelDNN.compile(loss=loss, optimizer=opti(lr=alpha))   # compile loaded model
""" ########################################################################## """

"""------------------------load and compile xgb------------------------------"""    

import xgboost as xgb

fileXGB = 'PP_4Marker_XGB/PP_XGB_4marker.model'

loaded_modelXGB = xgb.Booster({'nthread':1})   # init model

loaded_modelXGB.load_model(fileXGB)      # load data
""" ########################################################################## """

"""------------------------load and compile rfr------------------------------"""   
"""
import joblib

fileRFR = ''

loaded_model = joblib.load(load_rfr_path)
"""
""" ########################################################################## """

from keras.models import load_model

fileEDNN = 'PP_4Marker_eDNN/PP_eDNN_4marker_weights.h5'

loaded_modelEDNN  = load_model(fileEDNN)                   # load model from file


"""--------------------------load demo data (already normalized)------------------------------------"""

fileX = 'demoData/0_demoData_PP_4marker_X_testN.csv'
fileY = 'demoData/1_demoData_PP_Y_testN.csv'     

X_inN     = np.loadtxt(fileX) 
y_targetN = np.loadtxt(fileY) 


"""------------------predict values and make parity plot-----------------------"""

y_predN_DNN = loaded_modelDNN.predict(X_inN)

y_predN_XGB = loaded_modelXGB.predict(xgb.DMatrix(X_inN))


def predict_stacked_model(model, inputX):
	
	X = [inputX for _ in range(len(model.input))]      # prepare input data
	
	return model.predict(X, verbose=0)


y_predN_eDNN = predict_stacked_model(loaded_modelEDNN,X_inN)

xL  = np.arange(0.0, 0.8, 1e-3)
yL  = np.arange(0.0, 0.8, 1e-3)

plt.figure()
plt.title('normalized parity plot')
plt.plot(y_targetN,y_predN_DNN,'r.',label='DNN')
plt.plot(y_targetN,y_predN_XGB,'b.',label='XGB')
plt.plot(y_targetN,y_predN_eDNN,'g.',label='eDNN')
plt.plot(xL,yL,'k--')
plt.ylabel('prediction')
plt.xlabel('target')
plt.legend()

"""----------------- denormalize prediction, just shown for DNN-----------------------------------"""


xL  = np.arange(0.0, 0.2, 1e-3)
yL  = np.arange(0.0, 0.2, 1e-3)

y_pred   = minY+y_predN_DNN*(maxY-minY)
y_target = minY+y_targetN*(maxY-minY)

plt.figure()
plt.title('denormalized parity plot')
plt.plot(y_target,y_pred,'bo')
plt.plot(xL,yL,'k--')
plt.ylabel('prediction')
plt.xlabel('target')


