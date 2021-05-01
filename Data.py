import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from windrose import WindroseAxes

def data(winddatadf, powercurvedf):
    st.write('''## Data Visualisation''')
    st.write('It makes it simple to have a peek in your data')
    if winddatadf is not None and powercurvedf is not None:
        
        # Simple dataframe display
        cols = st.beta_columns(2)
        cols[0].write('''### winddata dataframe''')
        cols[0].write(winddatadf)
        cols[1].write('''### powerdata dataframe''')
        cols[1].write(powercurvedf)
        
        # PowerCurve Display
        st.write('''### Power Data plot''')

        # Only a subset of options make sense
        x_options = powercurvedf.columns
        y_options = powercurvedf.columns
        # Allow use to choose
        cols = st.beta_columns(2)
        x_axis = cols[0].selectbox('Choose Value for X axis', x_options)
        y_axis = cols[1].selectbox('Choose Value for y axis', y_options)
        # plot the value
        st.write(x_axis,' vs ', y_axis)
        fig = px.scatter(powercurvedf,
                        x=x_axis,
                        y= y_axis,
                        hover_name='Wind Speed (m/s)')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # WindRose
        st.write('''### Wind Rose Diagram''')
        
        fig1 = px.bar_polar(winddatadf, r="sped", theta="drct",
                           template="plotly_dark",
                           color_discrete_sequence= px.colors.sequential.Plasma_r)
        st.plotly_chart(fig1, use_container_width=True)
        
        
        
    else:
        st.error('File Not uploaded yet. Please upload the power and wind data first in order to visualize them')
        