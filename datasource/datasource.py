#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import nilmtk as nilmtk
from nilmtk import DataSet, MeterGroup, Appliance
from nilmtk.metergroup import MeterGroupID
from nilmtk.elecmeter import ElecMeter, ElecMeterID 
from typing import List, Tuple, Dict
from pandas import DataFrame
import numpy as np
import math


#get current working directory
cwd = os.getcwd()  
dirname = os.path.dirname(cwd)
REDD = os.path.join(dirname, r"datasource\dataset\REDD\redd.h5")
redd = DataSet(REDD)
year = '2011'
train_month_end = '5'
train_month_start = '4'
train_end_date = "{}-17-{}".format(train_month_end, year)
train_start_date = "{}-18-{}".format(train_month_start, year)
test_month_end = '5'
test_month_start = '5'
test_end_date = "{}-25-{}".format(test_month_end, year)
test_start_date = "{}-18-{}".format(test_month_start, year)
appliances_redd1 = ['fridge','dish washer','sockets','light','unknown','electric space heater','electric stove','electric oven',
                   'washer dryer']
building = 1


# In[ ]:


class Datasource():
    def __init__(self, dataset: DataSet, name:str):
        self.dataset = dataset
        self.name = name
    
    
    def get_selected_metergroup(self, building: int, appliances: List,  start: str, end: str, 
                               sample_period = 3, include_mains=True):
        """
        Get the MeterGroup of selected appliances
        Return the MeterGroup
        """
        self.dataset.set_window(start=start, end=end)
        elec = self.dataset.buildings[building].elec
        appliances_with_one_meter = []
        appliances_with_more_meters = []
        for appliance in appliances:
            metergroup = elec.select_using_appliances(type=appliances)
            if len(metergroup.meters) > 1:
                appliances_with_more_meters.append(appliance)
            else:
                appliances_with_one_meter.append(appliance)

        special_metergroup = None
        for appliance in appliances_with_more_meters:
            inst = 1
            if appliance == 'sockets' and building == 3:
                inst = 4
            if special_metergroup is None:
                special_metergroup = elec.select_using_appliances(type=appliance, instance=inst)
            else:
                special_metergroup = special_metergroup.union(elec.select_using_appliances(type=appliance, instance=1))

        selected_metergroup = elec.select_using_appliances(type=appliances_with_one_meter)
        selected_metergroup = selected_metergroup.union(special_metergroup)
        if include_mains:
            mains_meter = self.dataset.buildings[building].elec.mains()
            if isinstance(mains_meter, MeterGroup):
                if len(mains_meter.meters) > 1:
                    mains_meter = mains_meter.meters[0]
                    mains_metergroup = MeterGroup(meters=[mains_meter])
                else:
                    mains_metergroup = mains_meter
            else:
                mains_metergroup = MeterGroup(meters=[mains_meter])
            selected_metergroup = selected_metergroup.union(mains_metergroup)
        return selected_metergroup
    
    
    def read_selected_appliances(self, building: int, appliances : List, start: str, end: str, include_mains,
                               sample_period = 3):
        """
        Read and fill in missing values of selected appliances in a given bulding
        Return the DataFrame of selected appliances in which columns are ElecMeter IDs,
        and values are the PC of the meter in given sample period
        """
        selected_metergroup = self.get_selected_metergroup(building, appliances, start, end,sample_period,include_mains)
        df = selected_metergroup.dataframe_of_meters(sample_period=sample_period)
        df.fillna(0, inplace=True)
        return df, selected_metergroup
        
        
    def all_meters(self, building: int, start_date: str, end_date:str, sample_period = 3):
        """
        Read all the meters in given building
        """
        elec = self.dataset.buildings[building].elec
        print(elec)
        redd_datasource = Datasource(redd, "REDD")
        
        
    def get_labels_df(self, df: DataFrame, selected_metergroup: MeterGroup) -> [List, Dict]:
        """
        Returns two lists, one is a list of labels which describes DataFrame columns
        Other is a list of power threshold of each appliances
        """
        lst = []
        threshold = {}
        for m in df.columns:
        #print(m)
            label = ""
            if isinstance(m,MeterGroupID):
                tup_elecmeterID = m[0]
                lst_elecmeterID = list(tup_elecmeterID)
                #get meter group using list of ElecMeterIDs
                mg = selected_metergroup[lst_elecmeterID]
                #get labels of meter group
                labels = mg.get_labels(lst_elecmeterID)
                label = labels[0]
                threshold[label] =mg.on_power_threshold()
            else:
                #get ElecMeter using ElecMeterID
                elec_meter = selected_metergroup[m]
                #get labels of ElecMeter
                label = elec_meter.label()
                threshold[label] = elec_meter.on_power_threshold()
            lst += [label]
        print("Done making appliance labels from Data Frame")
        return lst, threshold

    def get_dic_real_power(self, df: DataFrame, labels: List) -> Dict:
        """
        Returns a Dictionary in which key is name of the appliance, and value is the power consumption the appliance
        """
        Dict = {}
        lst = []
        for k, v in df.items():
            lst += [v]
        
        for i in range(len(labels)):
            Dict[labels[i]] = lst[i]
            
        return Dict
    
    
    
    def get_dic_labeled_power(self, RealPower: Dict, threshold: Dict) -> Dict:
        """
        Returns a Dictionary in which key is name of the appliance, and value is 
        """
        LabeledPower = {}
        for appliance, real_power in RealPower.items():
            if appliance != 'Site meter':
                arr = self.create_labels(real_power, threshold[appliance])
                LabeledPower[appliance] = arr
        print("Done making Dictionary of labeled power")
        return LabeledPower
    
    
    def create_labels(self, array, threshold):
        res = np.empty(array.shape)
        for i in range(len(array)):
            if array[i] >= threshold:
                res[i] = 1
            else:
                res[i] = 0
        return list(res)
    
    
redd_datasource = Datasource(redd, "REDD")
#redd_datasource.all_meters(1, train_start_date, train_end_date, sample_period = 3)
#train dataset
train_df, train_selected_metergroup = redd_datasource.read_selected_appliances(1, appliances_redd1, train_start_date, train_end_date, True, sample_period = 3)
train_labels, train_threshold = redd_datasource.get_labels_df(train_df, train_selected_metergroup)
train_DictLabeled = redd_datasource.get_dic_real_power(train_df,train_labels)
train_site_meter = train_DictLabeled['Site meter']
train_labeled = redd_datasource.get_dic_labeled_power(train_DictLabeled,train_threshold)
#test dataset
test_df, test_selected_metergroup = redd_datasource.read_selected_appliances(1, appliances_redd1, test_start_date, test_end_date, True, sample_period = 3)
test_labels, test_threshold = redd_datasource.get_labels_df(test_df, test_selected_metergroup)
test_DictLabeled = redd_datasource.get_dic_real_power(test_df,test_labels)
test_site_meter = test_DictLabeled['Site meter']
test_labeled = redd_datasource.get_dic_labeled_power(test_DictLabeled,test_threshold)


# In[ ]:


train_labeled_fridge = train_labeled['Light'] 
test_labeled_fridge = test_labeled['Light'] 


# In[ ]:


import numpy as py
def dimension_handler_ndim(seconds : int, arr: List, sample_period):
    """
    Make batches of data with the given seconds and sample period
    """
    
    #take care of redundancies here
    new_arr = []
    rem = len(arr) % seconds
    if rem !=0:
        arr = arr[:-rem]
    n_sample = len(arr)//seconds
    print(n_sample)
    for i in range(n_sample):
        lst = list()
        for j in range(seconds):    
            lst.append([arr[(i*seconds)+j]])
        new_arr.append(lst)
        
    new_arr = np.array(new_arr)
    return new_arr

def dimension_handler_target(seconds : int, arr: List, sample_period):
    """
    Make batches of data with the given seconds and sample period
    """
    new_arr = list()
    
    rem = len(arr) % seconds
    if rem !=0:
        arr = arr[:-rem]
    n_sample = len(arr)//seconds
    for i in range(n_sample):
        lst = list()
        for j in range(seconds):    
            lst.append([arr[(i*seconds)+j]])
        new_arr.append(any(lst))
        
    new_arr = np.array(new_arr)
    
        
        
    
    
    return new_arr

window = 20
train_batches_X = dimension_handler_ndim(window, train_site_meter.values, 3) 
test_batches_X = dimension_handler_ndim(window, test_site_meter.values, 3)
train_batches_Y = dimension_handler_target(window, train_labeled_fridge, 3)
test_batches_Y = dimension_handler_target(window, test_labeled_fridge, 3)
print(f'Train: X.len = {len(train_batches_X)}  y.len = {len(train_batches_Y)}')
print(f'Valid: X.len = {len(test_batches_X)} Valid: y.len = {len(test_batches_Y)}')


# In[ ]:


import time
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

n_steps = 20
n_features = 1

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

train_batches_X = train_batches_X.reshape((train_batches_X.shape[0], train_batches_X.shape[1], 1))
start_time = time.time()
model.fit(train_batches_X, train_batches_Y, epochs=200, verbose=0)
test_batches_X = test_batches_X.reshape((test_batches_X.shape[0], test_batches_X.shape[1], 1))
print('Done Training {}'.format(round(time.time() - start_time, 2)))
yhat = model.predict(test_batches_X, verbose=0)
yhat_unpacked = [int(i) for i in yhat]
from sklearn.metrics import accuracy_score
print(accuracy_score(yhat_unpacked,test_batches_Y))

