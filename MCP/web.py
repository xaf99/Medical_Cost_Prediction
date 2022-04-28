from ast import FormattedValue
import numpy as np
import pickle as pkl
import streamlit as st
import pandas as pd
# from sklearn import svm,ensemble
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# #from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# #from sklearn.tree import DecisionTreeRegressor
# from sklearn.svm import SVR

file_name = "C:\\Users\\huzai\\Desktop\\MCP\\rf_tuned.pkl"
load_model = pkl.load(open(file_name, "rb"))

def prediction_data(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

    prediction = load_model.predict(input_data_reshape)
    return prediction

def main():
    st.title("Insurence Amount Prediction")

    age = st.text_input("Enter Age")
    sex = st.text_input("Enter gender (0 = Male, 1 = Female)")
    bmi = st.text_input("Enter BMI value")
    children = st.text_input("Enter number of Children")
    smoker = st.text_input("Enter smoking status (0 = No, 1 = Yes)")
    region = st.text_input("Enter Region (NorthWest = 0, Northeast = 1, SouthEast = 2, SouthWest = 3)")

    amount = ''

    if st.button("Test Result"):
        amount = prediction_data([age, sex, bmi, children, smoker, region])

    st.success(amount)
    
if __name__ == "__main__":
    main()