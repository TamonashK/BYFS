import streamlit as st
import numpy as np
import pandas as pd
import data
import plots
import predict

def app(cars_df):
	st.header('View Data')
	with st.expander('View Dataset'):
		st.table(cars_df)

	st.subheader("Columns Description:")
	if st.checkbox("Show summary"):
		st.table(cars_df.describe())   

	beta_col1, beta_col2 = st.columns(2)
	with beta_col1:
		if st.checkbox('Show All Column Names'):
			st.table(cars_df.columns)

	with beta_col2:
		if st.checkbox('View Column Data'):
			column_data = st.selectbox('Select Column', tuple(cars_df.columns))
			st.write(cars_df[column_data])
