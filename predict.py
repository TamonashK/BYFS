import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error, mean_squared_error

# Define the 'prediction()' function.
@st.cache
def prediction(car_df, carwidth, enginesize, horsepower, drivewheelfwd, carcompanybuick):
  x = car_df.iloc[:,:-1]
  y = car_df['price']

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

  lin_reg = LinearRegression()
  lin_reg.fit(x_train, y_train)
  score = lin_reg.score(x_train, y_train)
  price = lin_reg.predict([[carwidth, enginesize, horsepower, drivewheelfwd, carcompanybuick]])
  price = price[0]
  y_test_pred = lin_reg.predict(x_test)
  score_r2 = r2_score(y_test, y_test_pred)
  mae = mean_absolute_error(y_test, y_test_pred)
  msle = mean_squared_log_error(y_test, y_test_pred)
  rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
  return price, score, score_r2, mae, msle, rmse

def app(car_df):
  st.markdown("<p style='color:blue;font-size:25px'>This app uses <b>Linear regression</b> to predict the price of a car based on your inputs.", unsafe_allow_html = True) 
  st.subheader("Select Values:")
  carwidth = st.slider('Car Width', float(car_df['carwidth'].min()), float(car_df['carwidth'].max()))
  enginesize = st.slider('Engine Size', int(car_df['enginesize'].min()), int(car_df['enginesize'].max()))
  horsepower = st.slider('Horsepower', int(car_df['horsepower'].min()), int(car_df['horsepower'].max()))
  drw_fwd = st.radio('Is it a forward drive wheel car?', ('Yes', 'No'))
  if drw_fwd == 'No':
    drw_fwd = 0
  else:
    drw_fwd = 1
  cmpny = st.radio('Is the car manufactured by Buick', ('Yes', 'No'))
  if cmpny == 'No':
    cmpny = 0
  else:
    cmpny = 1
  if st.button('Predict'):
    st.subheader('Prediction Results')
    price, score, score_r2, mae, msle, rmse = prediction(car_df, carwidth, enginesize, horsepower, drw_fwd, cmpny)
    st.success("The predicted price of the car: ${:,}".format(int(price))) 
    st.info("Accuracy score of this model is: {:2.2%}".format(score)) 
    st.info(f"R-squared score of this model is: {score_r2:.3f}") 
    st.info(f"Mean absolute error of this model is: {mae:.3f}") 
    st.info(f"Mean squared log error of this model is: {msle:.3f}") 
    st.info(f"Root mean squared error of this model is: {rmse:.3f}")