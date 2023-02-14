# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 20:40:47 2023

@author: Nihal
"""

import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open("trained_model.sav",'rb'))

def heart_data_prediction(input_data):
    input_data=(53,1,0,140,203,1,0,155,1,3.1,0,0,3)
    input_numpy=np.asarray(input_data)
    input_data_reshape=input_numpy.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshape)
    if(prediction[0]==0):
      return 'The person doesnot have a heart disease'
    else:
      return 'The person has a heart disease'
  
    
  
def main():

    st.title('Heart disease Prediction web App')  
  
    age=st.text_input("Age")
    sex=st.text_input("Sex")
    cp=st.text_input("Cp")
    trestbps=st.text_input("trestbps")
    chol=st.text_input("chol")
    fbs=st.text_input("fbs")
    restecg=st.text_input("restecg")
    thalach=st.text_input("thalach")
    exang=st.text_input("exang")
    oldpeak=st.text_input("oldpeak")
    slope=st.text_input("slope")
    ca=st.text_input("ca")
    thal=st.text_input("thal")
    
    diagnosis=''
    
    if st.button('Heart disease test result'):
        diagnosis=heart_data_prediction([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
        
    st.success(diagnosis)
    
    
    
if __name__=='__main__':
    main()
