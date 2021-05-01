import pandas as pd
import numpy as np
import Data
import home
import optimizer
import streamlit as st

st.image('https://media1.giphy.com/media/OnavQV9tvZVks/giphy.gif', use_column_width=True)
st.title('Wind Farm Power Optimization')


st.sidebar.write('## upload wind and Power data') 
wind_data = st.sidebar.file_uploader(label =" ", type = ['csv', 'xlsx'])  
               
#st.sidebar.write('upload power data') 
powercurve_data = st.sidebar.file_uploader(label =" ", type = ['csv'])                 


# Visualizing Uploaded File
global winddatadf, powercurvedf

if wind_data is not None and powercurve_data is not None:
    st.sidebar.success("Files Uploaded successfully!")
    try:
        winddatadf = pd.read_csv(wind_data)
        powercurvedf = pd.read_csv(powercurve_data)
    except Exception as e:
        print(e)
        winddatadf = pd.read_excel(wind_data)
        powercurvedf = pd.read_excel(powercurve_data)
    
else:
    st.sidebar.write("No File Uploaded")      


# Sidebar Navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select a page:',  ['Home', 'Visualizer', 'Optimizer'])

if options == 'Home':
    home.home()
elif options == 'Visualizer':
    if wind_data is not None and powercurve_data is not None:
        Data.data(winddatadf, powercurvedf)
    else:
        Data.data(None, None)
elif options == 'Optimizer':
    if wind_data is not None and powercurve_data is not None:
        optimizer.optimizer(winddatadf, powercurvedf)
    else:
        optimizer.optimizer(None, None)



