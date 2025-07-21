import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

### Import data

dataset_choice = 1
dataset_dic = {1:'pure', 2:'single', 3:'multi'}
data_name = dataset_dic[dataset_choice]

user = 'ricky'

if user == 'ricky':
    path = '/home/rdube/PID_paper/data/'
    new_path = "/home/rdube/PID_paper/data_processed/"
elif user == "eric":
    path = '/projects/mccleary_group/habjan.e/PID_code/data_raw/'
    new_path = "/projects/mccleary_group/habjan.e/PID_code/data_processed/"

file = data_name + "Training_LE.hdf5"
filename = path + file
train = pd.read_hdf(filename, 'event1')

file = data_name + "Test_LE.hdf5"
filename = path + file
test = pd.read_hdf(filename, 'event1')

overflow_dict = {'E': -5, 'px':-500, 'py':-500, 'pz':-500, 'q':-10, 'E1E9':-5, 'E9E25': -5, 'docaTrack':-5, 'preshowerE':-5, 'sigLong': -5, 
            'sigTrans':-5, 'sigTheta':-5, 'E_L2':-5, 'E_L3':-5, 'E_L4':-5, 'dEdxCDC':-5, 'dEdxFDC':-5, 'tShower': -10, 'tShowerBCAL': -10, 'tShowerFCAL': -10, 
            'thetac':-5, 'bCalPathLength':-5, 'fCalPathLength':-5, 'dEdxTOF':-5, 'tofTOF':-5, 'pathLengthTOF':-5, 'dEdxSc': -5, 'pathLengthSc':-100, 
            'tofSc':-100, 'xShower': -500, 'yShower':-500, 'zShower':-500, 'xTrack':-500, 'yTrack':-500, 'zTrack':-500, 'CDChits':-5, 
            'FDChits':-5, 'DOCA':-5, 'deltaz':-100, 'deltaphi':-10 , 'tFlightSc':-10, 'tFlightBCAL':-10, 'tFlightTOF':-10, 'tFlightFCAL':-10}

## Training dataset is pretty minimal, so handle that first:

train_columns = ["ptype","px","py","pz","E","dEdxCDC","q"]
train['ptype'] = train['ptype'].astype(int)
train_charged = train[train['q'] != 0][train_columns]

### Test Dataset requires some unpacking:


test_columns  = ['px','py','pz','dEdxCDC','E','q','ptype','true ptype','group','tShowerBCAL','tShowerFCAL','tFlightBCAL','tFlightFCAL']
test[["tFlightBCAL",'tFlightTOF',"tFlightFCAL","tFlightSC"]] = pd.DataFrame(test['tFlights'].tolist(),index=test.index)

test['tShowerFCAL'] = np.where(test['E_L2'].isna(), test['tShower'], np.nan)
test['tShowerBCAL'] = np.where(test['E_L2'].notna(), test['tShower'], np.nan)


### Make ptype label intergers instead of strongs
test['ptype'] = test['ptype'].astype(int)
test['true ptype'] = test['true ptype'].astype(int)

### Fix docaTrack variable

test['docaTrack']=test['docaTrack'].replace(10**6, np.nan)

## Make sure we only use test data, not data that is being used for validation in our models

test = test[test['group']%40000>=20000]

test_charged = test[test['q'] != 0]
test_charged = test_charged[test_columns]

### Replace all nan values with a set overflow value

for label, overflow in overflow_dict.items():
    if label in train_columns:
        train_charged[label] = train_charged[label].fillna(overflow)
    if label in test_columns:
        test_charged[label] = test_charged[label].fillna(overflow)

### Save Edited datasets to 'data_processed' folder

file = data_name + "ManualTrain_LE_sorted_charged.hdf5"
filename = new_path + file
train_charged.to_hdf(filename, 'event1')

file = data_name + "ManualTest_LE_sorted_charged.hdf5"
filename = new_path + file
test_charged.to_hdf(filename, 'event1')