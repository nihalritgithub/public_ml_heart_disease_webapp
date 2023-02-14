# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle
loaded_model=pickle.load(open("trained_model.sav",'rb'))
input_data=(53,1,0,140,203,1,0,155,1,3.1,0,0,3)
input_numpy=np.asarray(input_data)
input_data_reshape=input_numpy.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshape)
if(prediction[0]==0):
  print('The person doesnot have a heart disease')
else:
  print('The person has a heart disease')
