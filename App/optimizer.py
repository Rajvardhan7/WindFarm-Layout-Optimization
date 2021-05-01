import pandas as pd
import numpy as np
import streamlit as st
import math
from utilityfunctions import loadPowerCurve, binWindResourceData, searchSorted, preProcessing, getAEP,  checkConstraints
from shapely.geometry import Point           # Imported for constraint checking
from shapely.geometry.polygon import Polygon
import plotly.express as px
from geneticalgorithm import geneticalgorithm as ga
import randomsearch as rs
import pyswarms as ps
import random

import warnings
warnings.filterwarnings("ignore")

def optimizer(wind_data, powercurve_data):
    st.write('''## Optimizer Result''')
    st.write('Uses Genetic Algorithm to converge to optimal x and y coordinates')
    if wind_data is not None and powercurve_data is not None :

        power_curve_data = loadPowerCurve(powercurve_data)
        st.success("Powerdata loaded successfully")
    
        wind_data =  binWindResourceData(wind_data)   
        st.success("winddata loaded sucessfully")
        
        # Turbine Specifications.
        st.write('''## Turbine Specifications''') 
        
        global turb_diam, turb_height
        turb_diam = st.number_input("Turbine Diameter (in m)",min_value= 60, max_value=120, value=100, step=1)
        turb_height = st.number_input("Turbine Height (in m)",min_value= 80, max_value=140, value=100, step=1)
        turb_specs    =  {   
                             'Name': 'Anon Name',
                             'Vendor': 'Anon Vendor',
                             'Type': 'Anon Type',
                             'Dia (m)': turb_diam,
                             'Rotor Area (m2)': 7853,
                             'Hub Height (m)': turb_height,
                             'Cut-in Wind Speed (m/s)': 3.5,
                             'Cut-out Wind Speed (m/s)': 25,
                             'Rated Wind Speed (m/s)': 15,
                             'Rated Power (MW)': 3
                         }
        
        turb_diam      =  turb_specs['Dia (m)']
        turb_rad       =  turb_diam/2 
        
        
        power_curve = power_curve_data
        wind_inst_freq = wind_data    
        st.write('''## Field Specifications''')
        global n
        n = st.number_input("Number of turbines, n",min_value= 10, max_value=60, value=40, step=1)
        side = st.slider("side length (in m)", min_value = 100, max_value = 10000, value = 4000) # in m , 100 - 10,000
        
        st.write('''## Constraints''')
        peri_constr = st.number_input("Perimeter constraint (in m)",min_value= 10, max_value=100, value=50, step=1) # 10 - 100
        prox_constr = st.number_input("Proximity constraint (in m)",min_value= 250, max_value=600, value=400, step=1) # 250-800
        
        n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(power_curve, n)
    
        
        st.write('''## Select the Algorithms to use''')
        
        if st.checkbox('Genetic Algorithm', value=False):
                col1, col2 = st.beta_columns([0.5, 9.5])
                col2.subheader("Using Genetic Algorithm for optimization")
                
                max_iter = col2.slider("Max Number of Iterations", min_value = 10, max_value = 1000, value = 50)
                population_size = col2.number_input("Population size",min_value= 10, max_value=100, value= 30, step=1)
                
                var_bound = np.array([[peri_constr,side - peri_constr]]*(2*n))
                
                algorithm_param = {'max_num_iteration':max_iter,\
                               'population_size':population_size,\
                               'mutation_probability':0.1,\
                               'elit_ratio': 0.2,\
                               'crossover_probability': 0.5,\
                               'parents_portion': 0.3,\
                               'crossover_type':'uniform',\
                               'max_iteration_without_improv':150}
                
                col2.write('If values are set click on run')
                if col2.button('Run'):
                    def f(z):
                        pen = 0
                        for i in range(n):
                            for j in range(i):
                                dist = math.sqrt((z[i]-z[j])**2+(z[n+i]-z[n+j])**2)
                                if dist>prox_constr:
                                   pen = pen + 600 + 1000*dist
                        data_dict = {'x':list(z[0:n]),'y':list(z[n:2*n])}
                        df1 = pd.DataFrame(data_dict)
                        global turb_coords_1
                        turb_coords_1 = df1.to_numpy(dtype = np.float32)
                        AEP = getAEP(turb_rad, turb_coords_1, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)
                        return (pen-AEP)
                
                    model = ga(function=f,dimension=n*2,variable_boundaries=var_bound,variable_type='real',algorithm_parameters=algorithm_param)
                    col2.write("model is running. please wait")
                    model.run()
            
                    checkConstraints(turb_coords_1, turb_diam)
                    col2.subheader('Optimized AEP obtained:')
                    col2.write(getAEP(turb_rad, turb_coords_1, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t))
                    col2.subheader('Optimal Coordinates')
                    col2.write(turb_coords_1)
                    
                    # Plot
                    col2.subheader('Field Plot')
                    fig = px.scatter(turb_coords_1, x = turb_coords_1[:, 0], y = turb_coords_1[:, 1])
                    col2.plotly_chart(fig, use_container_width=True)
                
        if st.checkbox('Random Search Algorithm', value=False):
                col1, col2 = st.beta_columns([0.5, 9.5])
                col2.subheader("Using RS for optimization")
                max_iter = col2.slider("Max Number of Iterations", min_value = 10, max_value = 1000, value = 50)
                
                col2.write('If values are set click on run')
                if col2.button('Run'):
                    def f(z):
                        pen = 0
                        for i in range(n):
                            for j in range(i):
                                dist = math.sqrt((z[i]-z[j])**2+(z[n+i]-z[n+j])**2)
                                if dist>prox_constr:
                                   pen = pen + 600 + 1000*dist
                        data_dict = {'x':list(z[0:n]),'y':list(z[n:2*n])}
                        df1 = pd.DataFrame(data_dict)
                        global turb_coords_2
                        turb_coords_2 = df1.to_numpy(dtype = np.float32)
                        AEP = getAEP(turb_rad, turb_coords_2, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)
                        return (AEP-pen)
                    
                    col2.write("model is running. please wait")
                    a, b = rs.optimize(function=f, dimensions=n*2, lower_boundary=np.array([peri_constr]*(n*2)), upper_boundary= np.array([side-peri_constr]*(n*2)), max_iter=1000, maximize=True)
                
                    col2.write('a:'); st.write(a)
                    data_dict = {'x':list(b[0:n]),'y':list(b[n:2*n])}
                    df1 = pd.DataFrame(data_dict)
                    global turb_coords_final
                    turb_coords_final = df1.to_numpy(dtype = np.float32)
               
                
                    checkConstraints(turb_coords_final, turb_diam)
                    col2.write('Optimized AEP obtained:')
                    col2.write(getAEP(turb_rad, turb_coords_final, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t))
                    col2.write('Optimal Coordinates')
                    col2.write(turb_coords_final)     
                    
                     # Plot
                    col2.subheader('Field Plot')
                    fig = px.scatter(turb_coords_final, x = turb_coords_final[:, 0], y = turb_coords_final[:, 1])
                    col2.plotly_chart(fig, use_container_width=True)
                        
        if st.checkbox('Particle Swarm Algorithm', value=False):
                
                col1, col2 = st.beta_columns([0.5, 9.5])
                col2.write("Using Particle Swarm for Optimization")
                max_iter = col2.slider("Max Number of Iterations", min_value = 10, max_value = 1000, value = 50)
                p = col2.number_input("P Normalization",min_value= 1, max_value=2, value= 2, step=1)
                k = col2.number_input("Number Of Neighbours",min_value= 1, max_value= n*2, value= 2, step=1)
                
                col2.write('If values are set click on run')
                if col2.button('Run'):
                    def f(z):
                        pen = 0
                        for i in range(n):
                            for j in range(i):
                                dist = math.sqrt((z[i]-z[j])**2+(z[n+i]-z[n+j])**2)
                                if dist>prox_constr:
                                   pen = pen + 600 + 1000*dist
                        data_dict = {'x':list(z[0:n]),'y':list(z[n:2*n])}
                        df1 = pd.DataFrame(data_dict)
                        global turb_coords_3
                        turb_coords_3 = df1.to_numpy(dtype = np.float32)
                        # print(z.shape)
                        AEP = getAEP(turb_rad, turb_coords_3, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)
                        return (pen-AEP)
            
                    temp = (np.array([peri_constr]),np.array([side-peri_constr]))
                
                    # Set-up hyperparameters
                    options = {'c1': 0.5, 'c2': 0.3, 'w':0.5, 'k': k, 'p': p}
                     
                    # Call instance of PSO
                    optimizer = ps.single.LocalBestPSO(n_particles=n*2, dimensions=1, options=options, bounds=temp)
                
                    # Perform optimization
                    col2.write("model is running. please wait")
                    cost, pos = optimizer.optimize(f, iters=1000)
                
                    checkConstraints(turb_coords_3, turb_diam)
                    col2.subheader('Optimized AEP obtained:')
                    col2.write(getAEP(turb_rad, turb_coords_3, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t))
                    col2.subheader('Optimal Coordinates')
                    col2.write(turb_coords_3)  
                    
                    # Plot
                    col2.subheader('Field Plot')
                    fig = px.scatter(turb_coords_3, x = turb_coords_3[:, 0], y = turb_coords_3[:, 1])
                    col2.plotly_chart(fig, use_container_width=True)
                    
        if st.checkbox('Greedy Search Algorithm', value=False):
            
            col1, col2 = st.beta_columns([0.5, 9.5])
            col2.write("Using Greedy Search for Optimization")
            max_iter = col2.slider("Max Number of Iterations", min_value = 10, max_value = 1000, value = 100)
                
            col2.write('If values are set click on run')
            if col2.button('Run'):
                x_new = np.zeros(n)
                y_new = np.zeros(n)
                iter=0
                val1 = (int)((int)(side/2)-prox_constr)
                val2 = (int)((int)(side/2)+prox_constr)
              
                
                if n%2==0:
                    
                    x = np.concatenate( (np.array([val1]*int(n/2)), np.array([val2]*int(n/2)) ))
                    y = np.concatenate(( np.array([val2]*int(n/2)),np.array([val1]*int(n/2)) ))
                else:
                    x = np.concatenate(np.array([val1]*(int)((n-1)/2)),np.array([val2]*(int)((n+1)/2)))
                    y = np.concatenate(np.array([val2]*(int)((n-1)/2)),np.array([val1]*(int)((n+1)/2)))
                
                data_dict = {'x':list(x),'y':list(y)}
                df1 = pd.DataFrame(data_dict)
                global turb_coords_4
              
                turb_coords_4 = df1.to_numpy(dtype = np.float32)
               
                #var_bound = np.array([[peri_constr,side-peri_constr]]*(2*n))
                while(iter<=max_iter):
                    for ind in range(n):
                        flagx = 0
                        flagy = 0
                        flagprox = 0
                        turb_coords1 = turb_coords_4
                     
                        aep1 = getAEP(turb_rad, turb_coords_4, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)
                        x_new[ind] = turb_coords_4[ind,0] + random.randint(-50,50)
                        y_new[ind] = turb_coords_4[ind,1] + random.randint(-50,50)
                        if x_new[ind]<(peri_constr) or x_new[ind]>(side-peri_constr):
                            flagx = 1
                        if y_new[ind]<(peri_constr) or y_new[ind]>(side-peri_constr):
                            flagy = 1
                        for ind2 in range(n):
                            if ((x_new[ind]-turb_coords_4[ind2,0])**2 + (y_new[ind]-turb_coords_4[ind2,1])**2) < prox_constr:
                                flagprox = 1
                        if flagprox == 0:
                            if flagx==0:
                                turb_coords_4[ind,0] = x_new[ind]
                            if flagy==0:
                                turb_coords_4[ind,1] = y_new[ind]
                        aep2 = getAEP(turb_rad, turb_coords_4, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)
                        if aep2<aep1:
                            turb_coords_4 = turb_coords1
                            
                        iter = iter + 1
            
            
                checkConstraints(turb_coords_4, turb_diam)
                col2.subheader('Optimized AEP obtained:')
                col2.write(getAEP(turb_rad, turb_coords_4, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t))
                col2.subheader('Optimal Coordinates')
                col2.write(turb_coords_4)  
                
                
                # Plot
                col2.subheader('Field Plot')
                fig = px.scatter(turb_coords_4, x = turb_coords_4[:, 0], y = turb_coords_4[:, 1])
                col2.plotly_chart(fig, use_container_width=True)
                    
    else:
        st.error('File Not uploaded yet. Please upload the power and wind data first in order to visualize them')