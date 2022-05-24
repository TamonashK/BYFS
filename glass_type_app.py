# Importing the necessary Python modules.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# Loading the dataset.
@st.cache()
def load_data():

    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']

@st.cache()
def prediction(model, ri, na, mg, al, si, k, ca, ba, fe):
  glasstype = model.predict([[ri, na, mg, al, si, k, ca, ba, fe]])
  glasstype = glasstype[0]
  if glasstype == 1:
    return "building windows float processed".upper()
  elif glasstype == 2:
    return "building windows non float processed".upper()
  elif glasstype == 3:
    return "vehicle windows float processed".upper()
  elif glasstype == 4:
    return "vehicle windows non float processed".upper()
  elif glasstype == 5:
    return "containers".upper()
  elif glasstype == 6:
    return "tableware".upper()
  else:
    return "headlamp".upper()

st.title('Glass Type prediction Web app')
st.sidebar.title('Glass Type prediction Web app')

if st.sidebar.checkbox('Show Raw Data'):
  st.subheader('Glass Type Data Set')
  st.dataframe(glass_df)

st.set_option('deprecation.showPyplotGlobalUse', False)

st.sidebar.subheader('Visualization Selector')
plotlist = st.sidebar.multiselect('Select the charts or plots', ('Correlation Heatmap', 'Line Chart', 'Area Chart', 'Count Plot', 'Box Plot', 'Pie Chart'))
if 'Correlation Heatmap' in plotlist:
  st.subheader('Correlation Heatmap')
  plt.figure(figsize = (9,5))
  sns.heatmap(glass_df.corr(), annot = True)
  st.pyplot()
if 'Count Plot' in plotlist:
  st.subheader('Count Plot')
  plt.figure(figsize = (9,5))
  sns.countplot(x = 'GlassType', data = glass_df)
  st.pyplot() 
if 'Line Chart' in plotlist:
  st.subheader('Line Chart')
  plt.figure(figsize = (9,5))
  st.line_chart(glass_df)
  st.pyplot() 
if 'Area Chart' in plotlist:
  st.subheader('Area Chart')
  plt.figure(figsize = (9,5))
  st.area_chart(glass_df)
  st.pyplot() 
if 'Box Plot' in plotlist:
  st.subheader('Box Plot')
  column = st.sidebar.selectbox('Select the Column for Box Plot', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))
  plt.figure(figsize = (9,5))
  sns.boxplot(glass_df[column])
  st.pyplot() 
if 'Pie Chart' in plotlist:
  st.subheader('Pie Chart')
  PieData = glass_df['GlassType'].value_counts()
  plt.figure(figsize = (9,5))
  plt.pie(PieData, labels = PieData.index, autopct = '%1.2f%%', startangle = 30, explode = np.linspace(.06, .12, 6))
  st.pyplot() 

st.sidebar.subheader('Select your Values')
ri = st.sidebar.slider('Input RI', float(glass_df['RI'].min()), float(glass_df['RI'].max()))
na = st.sidebar.slider('Input Na', float(glass_df['Na'].min()), float(glass_df['Na'].max()))
mg = st.sidebar.slider('Input Mg', float(glass_df['Mg'].min()), float(glass_df['Mg'].max()))
al = st.sidebar.slider('Input Al', float(glass_df['Al'].min()), float(glass_df['Al'].max()))
si = st.sidebar.slider('Input Si', float(glass_df['Si'].min()), float(glass_df['Si'].max()))
k = st.sidebar.slider('Input K', float(glass_df['K'].min()), float(glass_df['K'].max()))
ca = st.sidebar.slider('Input Ca', float(glass_df['Ca'].min()), float(glass_df['Ca'].max()))
ba = st.sidebar.slider('Input Ba', float(glass_df['Ba'].min()), float(glass_df['Ba'].max()))
fe = st.sidebar.slider('Input Fe', float(glass_df['Fe'].min()), float(glass_df['Fe'].max()))

st.sidebar.subheader('Choose Classifier')

classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Random Forest Classifier', 'Logistic Regression'))

from sklearn.metrics import plot_confusion_matrix
if classifier == 'Support Vector Machine':
  st.sidebar.subheader('Model Hyper Parameters')
  c_input = st.sidebar.number_input('C error rate', 1,100, step = 1)
  kernel_input = st.sidebar.radio('Kernel', ('linear', 'rbf', 'poly'))
  gamma_input = st.sidebar.number_input('Gamma', 1,100,step = 1)
  if st.sidebar.button('Classify'):
    st.subheader('Support Vector Machine')
    svc_model = SVC(C = c_input, kernel = kernel_input, gamma = gamma_input)
    svc_model.fit(X_train, y_train)
    y_pred = svc_model.predict(X_test)
    accuracy = svc_model.score(X_test, y_test)
    glasstype = prediction(svc_model, ri, na, mg, al, si, k, ca, ba, fe)
    st.write('The type of glass prediction is', glasstype)
    st.write('Accuracy:',accuracy)
    plot_confusion_matrix(svc_model, X_test, y_test)
    st.pyplot()

from sklearn.ensemble import RandomForestClassifier
if classifier == 'Random Forest Classifier':
  st.sidebar.subheader('Model Hyper Parameters')
  n_estimator_input = st.sidebar.number_input('Number of trees in forest', 100, 500, step = 10)
  maximum_depth_input = st.sidebar.number_input('Maximum depth of tree', 1, 100, step = 1)
  if st.sidebar.button('Classify'):
    st.subheader('Random Forest Classifier')
    rf_clf = RandomForestClassifier(n_estimators = n_estimator_input, max_depth = maximum_depth_input, n_jobs = -1)
    rf_clf.fit(X_train, y_train)
    accuracy = rf_clf.score(X_test, y_test)
    glass_type = prediction(rf_clf, ri, na, mg, al, si, k, ca, ba, fe)
    st.write('Accuracy is:',accuracy)
    st.write('Type of glass is:', glass_type)
    plot_confusion_matrix(rf_clf, X_test, y_test)
    st.pyplot()
 
from sklearn.linear_model import LogisticRegression    
if classifier == 'Logistic Regression':
  st.sidebar.subheader('Model Hyperparameters')
  c_value = st.sidebar.number_input('C (Error Rate)', 1, 100, step = 1)
  max_iter_input = st.sidebar.number_input('Max Iterations', 10, 10000, step = 10)
  
  if st.sidebar.button('Classify'):
    st.subheader('Logistic Regression')
    log_reg = LogisticRegression(C = c_value, max_iter = max_iter_input)
    log_reg.fit(X_train, y_train)
    score = log_reg.score(X_train, y_train)
    log_reg_pred = prediction(log_reg, ri, na, mg, al, si, k, ca, ba, fe)
    st.write('Accuracy:', score)
    st.write('Glass Prediction:', log_reg_pred)
    plot_confusion_matrix(log_reg, X_test, y_test)
    st.pyplot()