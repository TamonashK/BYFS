# S1.1: Design the "Visualise Data" page of the multipage app.
# Import necessary modules 
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Define a function 'app()' which accepts 'car_df' as an input.
def app(car_df):
  st.header('Visualize Data')
  st.set_option('deprecation.showPyplotGlobalUse', False)
  st.subheader('Scatter Plot')
  featurelist = st.multiselect('Select the X axis values', ('carwidth', 'enginesize', 'horsepower', 'drivewheel_fwd', 'car_company_buick'))
  for feature in featurelist:
    st.subheader(f'Scatterplot between {feature} and price')
    plt.figure(figsize = (7,5))
    sns.scatterplot(x = feature, y = 'price', data = car_df)
    st.pyplot()
  st.subheader('Visualization Selector')
  plottypes = st.multiselect('Select the Type of Plot', ('Histogram', 'Box Plot', 'Correlation Heatmap'))
  if 'Histogram' in plottypes:
    st.subheader('Histogram')
    column = st.selectbox('Select the Column to Create the Histogram', ('carwidth', 'enginesize', 'horsepower'))
    plt.figure(figsize = (7,5))
    plt.title(f'Histogram for {column}')
    plt.hist(car_df[column], bins = 'sturges', edgecolor = 'black')
    st.pyplot()
  if 'Box Plot' in plottypes:
    st.subheader('Box Plot')
    column = st.selectbox('Select the Column to Create the Box Plot', ('carwidth', 'enginesize', 'horsepower'))
    plt.figure(figsize =(7,3))
    plt.title(f'Box Plot for {column}')
    sns.boxplot(car_df[column])
    st.pyplot()
  if 'Correlation Heatmap' in plottypes:
    st.subheader('Correlation Heatmap')
    plt.figure(figsize = (7,5))
    ax = sns.heatmap(car_df.corr(), annot = True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    st.pyplot()