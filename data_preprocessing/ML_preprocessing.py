import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

### Import data

dataset_choice = 1
dataset_dic = {1:'pure', 2:'single', 3:'multi'}
data_name = dataset_dic[dataset_choice]

user = 'eric'

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

### Remove unwatned data labels and add a few more

listvals = ['tFlights', 'xTOF', 'yTOF', 'zTOF']

### Remove unwated labels
train = train.drop(listvals, axis=1)
test = test.drop(listvals, axis=1)

### Make ptype label intergers instead of strongs
train['ptype'] = train['ptype'].astype(np.int64)
test['true ptype'] = test['true ptype'].astype(np.int64)

### Fix docaTrack variable

test['docaTrack'] = test['docaTrack'].replace(10**6, np.nan)
train['docaTrack']= train['docaTrack'].replace(10**6, np.nan)

## Split test dataset into test and val

val = test[test['group']%40000<20000]
test = test[test['group']%40000>=20000]

### Seprate datasets by charged or neutral particles


### Replace all nan values with a set overflow value


rep_dict = {'E': -5, 'px':-500, 'py':-500, 'pz':-500, 'q':-10, 'E1E9':-5, 'E9E25': -5, 'docaTrack':-5, 'preshowerE':-5, 'sigLong': -5, 
            'sigTrans':-5, 'sigTheta':-5, 'E_L2':-5, 'E_L3':-5, 'E_L4':-5, 'dEdxCDC':-5, 'dEdxFDC':-5, 'tShower': -10, 
            'thetac':-5, 'bCalPathLength':-5, 'fCalPathLength':-5, 'dEdxTOF':-5, 'tofTOF':-5, 'pathLengthTOF':-5, 'dEdxSc': -5, 'pathLengthSc':-100, 
            'tofSc':-100, 'xShower': -500, 'yShower':-500, 'zShower':-500, 'xTrack':-500, 'yTrack':-500, 'zTrack':-500, 'CDChits':-5, 
            'FDChits':-5, 'DOCA':-5, 'deltaz':-100, 'deltaphi':-10 }


for label, overflow in rep_dict.items():
    train[label] = train[label].fillna(overflow)
    test[label] = test[label].fillna(overflow)
    val[label] = val[label].fillna(overflow)


### Charged datasets
test_char = test[test['q'] != 0]
train_char = train[train['q'] != 0]
val_char = val[val['q'] != 0]

### Neutral datasets
test_neut = test[test['q'] == 0].drop(columns=['px', 'py', 'pz', 'q', 'dEdxCDC', 'dEdxFDC', 'thetac', 'bCalPathLength', 'fCalPathLength', 'dEdxTOF', 'tofTOF', 'pathLengthTOF', 'dEdxSc', 'pathLengthSc', 'tofSc', 'xTrack', 'yTrack', 'zTrack', 'CDChits', 'FDChits', 'DOCA', 'deltaz', 'deltaphi'])
train_neut = train[train['q'] == 0].drop(columns=['px', 'py', 'pz', 'q', 'dEdxCDC', 'dEdxFDC', 'thetac', 'bCalPathLength', 'fCalPathLength', 'dEdxTOF', 'tofTOF', 'pathLengthTOF', 'dEdxSc', 'pathLengthSc', 'tofSc', 'xTrack', 'yTrack', 'zTrack', 'CDChits', 'FDChits', 'DOCA', 'deltaz', 'deltaphi'])
val_neut = val[val['q'] == 0].drop(columns=['px', 'py', 'pz', 'q', 'dEdxCDC', 'dEdxFDC', 'thetac', 'bCalPathLength', 'fCalPathLength', 'dEdxTOF', 'tofTOF', 'pathLengthTOF', 'dEdxSc', 'pathLengthSc', 'tofSc', 'xTrack', 'yTrack', 'zTrack', 'CDChits', 'FDChits', 'DOCA', 'deltaz', 'deltaphi'])

### Save Edited datasets to 'data_processed' folder

### Save charged datasets
file = data_name + "Training_LE_sorted_charged.hdf5"
filename = new_path + file
train_char.to_hdf(filename, 'event1', complevel=9)

file = data_name + "Test_LE_sorted_charged.hdf5"
filename = new_path + file
test_char.to_hdf(filename, 'event1', complevel=9)

file = data_name + "Val_LE_sorted_charged.hdf5"
filename = new_path + file
val_char.to_hdf(filename, 'event1', complevel=9)


### Save Neutral datsets
file = data_name + "Training_LE_sorted_neutral.hdf5"
filename = new_path + file
train_neut.to_hdf(filename, 'event1', complevel=9)

file = data_name + "Test_LE_sorted_neutral.hdf5"
filename = new_path + file
test_neut.to_hdf(filename, 'event1', complevel=9)

file = data_name + "Val_LE_sorted_neutral.hdf5"
filename = new_path + file
val_neut.to_hdf(filename, 'event1', complevel=9)
