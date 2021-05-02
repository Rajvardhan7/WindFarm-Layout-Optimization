import pandas as pd
import numpy as np
import streamlit as st


def home():
 
    st.info("""Give us the wind farm data and we will tell you the optimal locations
         for wind turbine Placement""")

    st.write('### Introdction to the interface')
    
    st.write('''The webapp is an interface to optimize the layout of a given wind farm. 
             The key problem with an unoptimized layout is the combined influence of arranged
             wind turbines on the wind speed distribution across the limited area of farm.''')
    st.write('To begin using the app, load your LAS file using the file upload option on the sidebar. Once you have done this, you can navigate to the relevant tools using the Navigation menu.')
    st.write('\n')
    st.write('### Sections')
    st.write('**Home:** A brief description of the web app.')
    st.write('**Visualizer:** for data visualization')
    st.write('**Optimizer:** Using certain algorithms and utility functions gives the optimal placement coordinates of the turbines. Also gives optimal AEP')
    st.write('\n')
    st.write('##           - Created by Team Energy Optimizers)
