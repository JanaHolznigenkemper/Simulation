#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd 
import time
import datetime
import numpy as np
import random
import copy
import pickle


# ## Production Times 

# Preparing analysed production times from MDC data of the past five years.  
# 'production_data_anonymized.csv' contains the analysed MDC data for production times  

# In[36]:


production_times_csv = pd.read_csv('production_data_anonymised.csv', sep=',')
materials = pd.read_csv('Material_anonymised.csv', sep= ';', usecols =['MATNR', 'COMNR'])
materials = materials.rename(columns = {'MATNR':'operation'})

#join to get information about commission numbers 
production_times_join = pd.merge(materials, production_times_csv, how ="inner", on = ['operation'])

# probabilities must sum to 1 (Rounding errors)
for i in range(0,len(production_times_join)):
    probs = production_times_join.iloc[i,3:]
    summe = sum(probs)
    if (summe !=1):
        difference = 1 - summe 
        index_maximum = list(probs).index(max(list(probs)))
        production_times_join.iloc[i,index_maximum+3] = production_times_join.iloc[i,index_maximum+3] + difference
        

#get production time values from column string 
intervals_string = production_times_csv.columns
production_time_values = []
for i in range(2,len(intervals_string)): 
    string_split = intervals_string[i].split()
    tuple_1 = float(string_split[2])
    tuple_2 = float(string_split[0])
    tuple_all = (tuple_1,tuple_2)
    production_time_values.append(tuple_all)

# take mean value of each interval of production times = production times that are possible on transfer presses (strokes per min)    
production_time_values_one = []
for i in range(0,len(production_time_values)):
    production_time_values_one.append(np.mean([production_time_values[i][1],production_time_values[i][0]]))
        


# ## Disruption Times 

# Preparing analysed disruption times from MDC data of the past five years.  
# "disruption_data_anonymized.csv" contains the analysed MDC data for disruption times. Downtimes on weekends are not included. The different diruption intervals vary in length accoring to an exponential transformation   

# In[37]:


disruption_times_csv_new = pd.read_csv("disruption_data_anonymised.csv", sep=",")
disruption_times_csv_new = disruption_times_csv_new.rename(columns = {"no_prod/contract_time" : "disruption"})
disruption_times_csv_new["no_disruption"] = 1 - disruption_times_csv_new["disruption"]

# get information about commission numbers 
disruption_times_join = pd.merge(materials , disruption_times_csv_new, how ="inner", on = ["operation"])

## probabilities must sum to 1 (rounding errors)
for i in range(0,len(disruption_times_join)):
    probs = disruption_times_join.iloc[i,3:(len(disruption_times_join.columns)-2)]
    summe = sum(probs)
    if (summe !=1):
        difference = 1 - summe 
        index_maximum = list(probs).index(max(list(probs)))
        disruption_times_join.iloc[i,index_maximum+3] = disruption_times_join.iloc[i,index_maximum+3] + difference
        


# In[38]:


#Convert time for disruption intervals 

disruption_intervals = [] 
disruption_columns = disruption_times_join.columns

# add first interval manually, the points in time are shifted backwards 
disruption_intervals.append((60,419))
for i in range(3,(len(disruption_columns)-3)):
    bound_left = int(disruption_columns[i]) * 60 
    bound_right = int(disruption_columns[i+1]) *60 
    interval = (bound_left, bound_right-1)
    disruption_intervals.append(interval)


# ### Transform csv with probability to random-roulette-lists (for speed-up) 

# In[39]:


production_times_roulette ={}
for i,v in production_times_join.iterrows():
    key = (v[0],v[2])
    z= zip(production_time_values_one, v[3:])
    temp_list = []
    for j,k in z:
        new_values = [j] * int(k* 10000)
        temp_list.extend(new_values)
    production_times_roulette[key] = temp_list


# In[40]:


disruption_times_decision_data = disruption_times_join[['operation', 'COMNR', 'machine','disruption', 'no_disruption']].copy()
disruption_times_decision_data['disruption'] = np.round(disruption_times_decision_data['disruption'],4) 
disruption_times_decision_data['no_disruption'] = np.round(disruption_times_decision_data['no_disruption'],4) 


# In[41]:


disruption_decision_roulette = {}
for i,v in disruption_times_decision_data.iterrows():
    key = (v[0],v[2])
    #disruption case = 1 
    z= zip([1,0], v[3:])
    temp_list = []
    for j,k in z:
        new_values = [j] * int(k* 10000)
        temp_list.extend(new_values)
    disruption_decision_roulette[key] = temp_list


# In[42]:


disruption_times_roulette = {}
length_intervals = len(disruption_intervals)
length = len(disruption_times_join.columns)
values = list(range(0,length_intervals))
for i,v in disruption_times_join.iterrows():
    key = (v[0],v[2])
    z= zip(values , v[3:length-3])
    temp_list = []
    for j,k in z:
        new_values = [j] * int(k* 10000)
        temp_list.extend(new_values)
    disruption_times_roulette[key] = temp_list


# ## Functions for Calculating Production and Disruption Time from underlying Distributions

# In[43]:


""" get random production time for a machine-material = machine-operation combinatin. Here all random production times per
part are summed up 
Parameters
----------
machine : int
    Number of the regarded machine 
material: int
    Number of the regarded material/operation
quantity: int 
    Number of parts to be processed
seed: int 
    We hand over a random seed to ensure randomness in parallelisation

Returns
-------
timedelta
    disruption time 
"""
def get_random_production_time_roulette(machine, material, quantity, seed):
    np.random.seed(seed)
    return pd.to_timedelta(np.sum(np.random.choice(production_times_roulette[(material, machine)],quantity)), unit = 's')


# In[44]:


""" get random disruption decision for a machine-material = machine-operation combination. If no disription occurs timedelta of 0 is returned
otherwise we get a random disruption time 
Parameters
----------
machine : int
    Number of the regarded machine 
material: int
    Number of the regarded material/operation
seed: int 
    We hand over a random seed to ensure randomness in parallelisation

Returns
-------
timedelta
    disruption time 
"""
def disruption_decision_r(machine, material, seed):
    np.random.seed(seed)
    decision = np.random.choice(disruption_decision_roulette[(material, machine)])
    if decision ==1:
        return get_random_disruption_time_r(machine, material,seed)
    else:
        return pd.to_timedelta(0, unit = 's') 


# In[45]:


""" get random disruption time for a machine-material = machine-operation combination 
Parameters
----------
machine : int
    Number of the regarded machine 
material: int
    Number of the regarded material/operation
seed: int 
    We hand over a random seed to ensure randomness in parallelisation

Returns
-------
timedelta
    disruption time 
"""

def get_random_disruption_time_r(machine, material, seed): 
    np.random.seed(seed)
    disruption_index = np.random.choice(disruption_times_roulette[(material, machine)])
    # first the disruption time intervals is determined
    d_interval = disruption_intervals[disruption_index]
    # for an exact disruption value, chose value from this interval randomly 
    disruption_time = np.random.choice(list(range(d_interval[0], d_interval[1]))) 
    
    
    return pd.to_timedelta(disruption_time, unit = 's')


# # Simulation

# #### Simulate a given schedule: For each operation production and disruption times are determined 

# 0start_planned 	1end_planned 	2machine 	3material_com 	4material_num 	5production_time 	6setup_time 	7quantity 	
# 8material_sim 	9machine_sim 	10disruption_buffer 	11production_time_random 	12disruption_random 	13start_new 	14end_new

# Simulation: for each part of the operation a random time is chosen from production distribution.  
# For each opertion it is decided once, if a disruption occurs

# In[1]:


""" Simulate schedule 
Parameters
----------
test_set_input : dict
    Contains all jobs scheduled on all levels. Structure of the dict:Level:dict:Machines:Df

Returns
-------
dict
    same structure as test_set_input, enriched with new end and start dates 
"""
def simulation(test_set_input, seed_list):
    test_set = copy.deepcopy(test_set_input)
# for each job on each level random production and disruption times are drawn
    for index_level, level_dict in test_set.items():
        seed = seed_list[index_level-1]
        for index_machine,df_machines in level_dict.items():
            for i, (index_df, value) in enumerate(df_machines.iterrows()):
                value['production_time_random'] =  get_random_production_time_roulette(value['machine_sim'], value['material_sim'], value['quantity'], seed)
                value['disruption_random'] = disruption_decision_r(value['machine_sim'], value['material_sim'], seed)
                test_set[index_level][index_machine].loc[index_df,'production_time_random'] = value['production_time_random']
                if (index_level == 1 and i==0):
                    start_new = value['start_planned']
                elif (index_level>1 and i==0):
                    start_new = max(df_machines.iloc[0,0], test_set[index_level-1][index_machine].iloc[0,14]) 
                elif (index_level==1 and i>0):
                    start_new = max(value["start_planned"], df_machines.iloc[i-1 ,13])
                elif (index_level>1 and i>0):
                    start_new = max(value["start_planned"], df_machines["end_new"].iloc[i-1], test_set[index_level-1][index_machine].iloc[i, 14])                           
                else:
                    print('something went wrong')
                test_set[index_level][index_machine].iloc[i,13] = start_new
                test_set[index_level][index_machine].iloc[i,14] = start_new + value["production_time_random"] + value["setup_time"] + value["disruption_random"]
        

    return test_set

