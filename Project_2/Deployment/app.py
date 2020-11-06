# -*- coding: utf-8 -*-
"""
@author: Raushan
"""

import streamlit as st
import pickle
import numpy as np

# Loading the Pickel Model
model = pickle.load(open(r"rf.pkl", "rb"))


def predict_default(features):

    features = np.array(features).astype(np.float64).reshape(1,-1)
    
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    return prediction, probability


def main():

    html_temp = """
        <div style = "background-color:tomato; padding: 10px;">
            <center><h1>Credit Default Predictor ML App</h1></center>
        </div><br>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    LIMIT_BAL = st.text_input("Limit Balance (in $)")
    
    education_status = ["Graduate School", "University", "High School", "Others"]
    marital_status = ["Married","Single", "others"]

    payment_status = [
        "Account started that month with a zero balance, and never used any credit.",
        "Account had a balance that was paid in full.",
        "At least the minimum payment was made, but the entire balance wasn't paid.",
        "Payment delay for 1 month.",
        "Payment delay for 2 month.",
        "Payment delay for 3 month.",
        "Payment delay for 4 month.",
        "Payment delay for 5 month.",
        "Payment delay for 6 month.",
        "Payment delay for 7 month.",
        "Payment delay for 8 month.",   
    ]

    EDUCATION = education_status.index(st.selectbox(
        "Select Education",
        tuple(education_status)
    )) + 1
    
    MARRIAGE = marital_status.index(st.selectbox(
        "Marital Status",
        tuple(marital_status)
    )) + 1
    
    AGE = st.text_input("Age (in Years)")

    PAY_1 = payment_status.index(st.selectbox(
        "Last Month Payment Status",
        tuple(payment_status)
    )) - 2
     
    BILL_AMT1 = st.text_input("Last month Bill Amount ( in (NT) $ )")
    BILL_AMT2 = st.text_input("Last 2nd month Bill Amount ( in (NT) $ )")
    BILL_AMT3 = st.text_input("Last 3rd month Bill Amount ( in (NT) $ )")
    BILL_AMT4 = st.text_input("Last 4th month Bill Amount ( in (NT) $ )")
    BILL_AMT5 = st.text_input("Last 5th month Bill Amount ( in (NT) $ )")
    BILL_AMT6 = st.text_input("Last 6th month Bill Amount ( in (NT) $ )")

    PAY_AMT1 = st.text_input("Amount paid in Last Month ( in (NT) $)")
    PAY_AMT2 = st.text_input("Amount paid in Last 2nd month ( in (NT) $ )")
    PAY_AMT3 = st.text_input("Amount paid in Last 3rd month ( in (NT) $ )")
    PAY_AMT4 = st.text_input("Amount paid in Last 4th month ( in (NT) $ )")
    PAY_AMT5 = st.text_input("Amount paid in Last 5th month ( in (NT) $ )")
    PAY_AMT6 = st.text_input("Amount paid in Last 6th month ( in (NT) $ )")

    if st.button("Predict"):
        
        features = [LIMIT_BAL,EDUCATION,MARRIAGE,AGE,PAY_1,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6]
        prediction, probability = predict_default(features)
        if prediction[0] == 1:
            st.success('The account is highly probable to default.')
            st.success("This account will default with a probability of {}%.".format(round(np.max(probability)*100, 2)))
        else:
            st.success('Credit can be lended.')
            st.success("This account will not be default with a probability of {}%.".format(round(np.max(probability)*100, 2)))

    if st.button("About"):
        st.text("Predicts the defaulter and prevent losses to the lender.")
        st.text("Built by Raushan")
        
if __name__ == '__main__':
    main()
