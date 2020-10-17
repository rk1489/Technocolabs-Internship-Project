# -*- coding: utf-8 -*-
"""
@author: Raushan
"""
import numpy as np
import pickle
import pandas as pd
import streamlit as st 

pickle_in=open('rf.pkl','rb')
rf=pickle.load(pickle_in)
    
def predict_default(limit_bal,education,marriage,age,pay_1,bill_amt1,
                                bill_amt2,bill_amt3,bill_amt4,bill_amt5,
                                bill_amt6,pay_amt1,pay_amt2,pay_amt3,
                                pay_amt4,pay_amt5,pay_amt6):
    
    """
    parameters:  
      - name: limit_bal
        in: query
        type: number
        required: true
      - name: education
        in: query
        type: number
        required: true
      - name: marriage
        in: query
        type: number
        required: true
      - name: age
        in: query
        type: number
        required: true
      - name: pay_1
        in: query
        type: number
        required: true
      - name: bill_amt1
        in: query
        type: number
        required: true
      - name: bill_amt2
        in: query
        type: number
        required: true
      - name: bill_amt3
        in: query
        type: number
        required: true
      - name: bill_amt4
        in: query
        type: number
        required: true
      - name: bill_amt5
        in: query
        type: number
        required: true
      - name: bill_amt6
        in: query
        type: number
        required: true
      - name: pay_amt1
        in: query
        type: number
        required: true
      - name: pay_amt2
        in: query
        type: number
        required: true
      - name: pay_amt3
        in: query
        type: number
        required: true
      - name: pay_amt4
        in: query
        type: number
        required: true
      - name: pay_amt5
        in: query
        type: number
        required: true
      - name: pay_amt6
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=rf.predict([[limit_bal,education,marriage,age,pay_1,bill_amt1,
                                bill_amt2,bill_amt3,bill_amt4,bill_amt5,
                                bill_amt6,pay_amt1,pay_amt2,pay_amt3,
                                pay_amt4,pay_amt5,pay_amt6]])
    print(prediction)
    return prediction


def main():
    st.title("Credit Default Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Credit Default Predictor ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    pay_1 = st.text_input("PAY_1","Type Here")
    limit_bal = st.text_input("LIMIT_BAL","Type Here")
    age = st.text_input("Age","Type Here")
    education = st.text_input("Education","Type Here")
    marriage = st.text_input("Marriage","Type Here")
    pay_amt1 = st.text_input("PAY_AMT1","Type Here")
    pay_amt2 = st.text_input("PAY_AMT2","Type Here")
    pay_amt3 = st.text_input("PAY_AMT3","Type Here")
    pay_amt4 = st.text_input("PAY_AMT4","Type Here")
    pay_amt5 = st.text_input("PAY_AMT5","Type Here")
    pay_amt6 = st.text_input("PAY_AMT6","Type Here")
    bill_amt1 = st.text_input("BILL_AMT1","Type Here")
    bill_amt2 = st.text_input("BILL_AMT2","Type Here")
    bill_amt3 = st.text_input("BILL_AMT3","Type Here")
    bill_amt4 = st.text_input("BILL_AMT4","Type Here")
    bill_amt5 = st.text_input("BILL_AMT5","Type Here")
    bill_amt6 = st.text_input("BILL_AMT6","Type Here")
    result=""
    if st.button("Predict"): 
        result=predict_default(limit_bal,education,marriage,age,pay_1,bill_amt1,
                                bill_amt2,bill_amt3,bill_amt4,bill_amt5,
                                bill_amt6,pay_amt1,pay_amt2,pay_amt3,
                                pay_amt4,pay_amt5,pay_amt6)
        pos_proba=rf.predict_proba([[limit_bal,education,marriage,age,pay_1,bill_amt1,
                                bill_amt2,bill_amt3,bill_amt4,bill_amt5,
                                bill_amt6,pay_amt1,pay_amt2,pay_amt3,
                                pay_amt4,pay_amt5,pay_amt6]])
        if result:
            st.success('The person is highly probable to default.')
            st.success('Probability to default = {}'.format(pos_proba[0][1]))
        else:
            st.success('Credit can be lended.')
            st.success('Probability to default = {}'.format(pos_proba[0][1]))
     
    if st.button("About"):
        st.text("Predicts the defaulter and prevent losses to the lender.")
        st.text("Built by Raushan")

if __name__=='__main__':
    main()