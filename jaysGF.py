# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 20:17:19 2021

@author: alexl
"""


import numpy as np
import pandas as pd

data = pd.read_csv(r'C:\Users\alexl\OneDrive\Documents\GitHub\cryptopred\depression.csv')

data = data[data['Indicator'] == 'Symptoms of Depressive Disorder']


data = data.groupby('Time_Period').Value.mean()

print(data.max())