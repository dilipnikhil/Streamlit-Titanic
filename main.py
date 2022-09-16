import streamlit as st
import pandas as pd
from utils import PrepProcesor,columns

import numpy as np
import joblib

model = joblib.load("xgbpipe.joblib")


st.title("Did they survive? :ship:")
passengerid = st.text_input('Input a passenger id' , "123456")
passengerClass = st.select_slider('Choose passenger class',[1,2,3])
name = st.text_input("Inout the passenger name",'Dilip Nikhil F')
gender = st.select_slider("Select Gender",["male","female"])
age = st.slider("Input age",0,100)
sibsp = st.slider("Input siblings",0,10)
parch = st.slider('Input parents/children',0,2)
ticketid =st.number_input("Ticket number",12345)
fare =st.number_input('Fare amount',0,100)
cabin =st.text_input("Enter cabin",'C53')
embarked =st.selectbox('choose embarkation point',["S","C","Q"])

#columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin', 'Embarked']

def predict():
    row=np.array([passengerid,passengerClass,name,gender,age,sibsp,parch,ticketid,fare,cabin,embarked])
    X = pd.DataFrame([row],columns=columns)
    prediction = model.predict(X)[0]

    if prediction ==1:
        st.success ("They survived, but Dicaprio did not !!")
    else:
        st.error( " This guy did not survive, guess he was DiCaprio !!")



st.button('Predict',on_click=predict)