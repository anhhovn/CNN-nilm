#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import nilmtk as nilmtk
from nilmtk import DataSet, MeterGroup, Appliance
from nilmtk.metergroup import MeterGroupID
from nilmtk.elecmeter import ElecMeter, ElecMeterID 
from typing import List, Tuple
from pandas import DataFrame


#get current working directory
cwd = os.getcwd()  
dirname = os.path.dirname(cwd)
REDD = os.path.join(dirname, r"datasource\dataset\REDD\redd.h5")
redd = DataSet(REDD)
year = '2011'
month_end = '8'
month_start = '1'
end_date = "{}-30-{}".format(month_end, year)
start_date = "{}-1-{}".format(month_start, year)
appliances_redd1 = ['washer dryer','electric oven', 'fridge', 'microwave', 'dish washer', 'unknown', 'sockets', 'light', 'electric space heater', 'electric stove']
building = 1


# In[165]:


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
        redd_datasource = Datasource(redd, "REDD")
        
        
    def get_labels_df(self, df: DataFrame, selected_metergroup: MeterGroup):
        """
        Returns a list of labels which describes DataFrame columns
        """
        lst = []
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
            else:
                #get ElecMeter using ElecMeterID
                elec_meter = selected_metergroup[m]
                #get labels of ElecMeter
                label = selected_metergroup[m].label()
                lst += [label]
        return lst

redd_datasource = Datasource(redd, "REDD")
redd_datasource.all_meters(1, start_date, end_date, sample_period = 3)
df, selected_metergroup = redd_datasource.read_selected_appliances(1, appliances_redd1, start_date, end_date, True, sample_period = 3)
labels = redd_datasource.get_labels_df(df, selected_metergroup)
print(labels)

