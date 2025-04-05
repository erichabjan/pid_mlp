import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

### Import data

dataset_choice = 1
dataset_dic = {1:'pure', 2:'single', 3:'multi'}
data_name = dataset_dic[dataset_choice]

path = '/Users/erich/Downloads/UConn/Undergraduate-Research/PID_code/data_raw/'

file = data_name + "Training_LE.hdf5"
filename = path + file
train = pd.read_hdf(filename, 'event1')

file = data_name + "Test_LE.hdf5"
filename = path + file
test = pd.read_hdf(filename, 'event1')

### Remove unwatned data labels and add a few more

listvals = ['tFlights', 'xTOF', 'yTOF', 'zTOF']

### Add BCAL calculated times
tbcal_te = np.array(test['tFlights'].to_list())[:, 0]
tbcal_tr = np.array(train['tFlights'].to_list())[:, 0]
train = train.assign(tFlightBCAL=tbcal_tr)
test = test.assign(tFlightBCAL=tbcal_te)

### Add TOF calculated times
ttof_te = np.array(test['tFlights'].to_list())[:, 1]
ttof_tr = np.array(train['tFlights'].to_list())[:, 1]
train = train.assign(tFlightTOF=ttof_tr)
test = test.assign(tFlightTOF=ttof_te)

### Add FCAL calculated times
tfcal_te = np.array(test['tFlights'].to_list())[:, 2]
tfcal_tr = np.array(train['tFlights'].to_list())[:, 2]
train = train.assign(tFlightFCAL=tfcal_tr)
test = test.assign(tFlightFCAL=tfcal_te)

### Add SC calculated times
tsc_te = np.array(test['tFlights'].to_list())[:, 3]
tsc_tr = np.array(train['tFlights'].to_list())[:, 3]
train = train.assign(tFlightSc=tsc_tr)
test = test.assign(tFlightSc=tsc_te)

### Make separate tShower BCAL and FCAL times, then add to dataframe
tshower_bcal_tr = np.full(len(train), np.nan)
tshower_bcal_te = np.full(len(test), np.nan)
tshower_fcal_tr = np.full(len(train), np.nan)
tshower_fcal_te = np.full(len(test), np.nan)

tShower_tr = np.array(train['tShower'])
tShower_te = np.array(test['tShower'])

bcal_bool_tr = ~np.isnan(np.array(train['E_L2']))
bcal_bool_te = ~np.isnan(np.array(test['E_L2']))
fcal_bool_tr = ~np.isnan(np.array(train['E1E9']))
fcal_bool_te = ~np.isnan(np.array(test['E1E9']))

tshower_bcal_tr[bcal_bool_tr] = tShower_tr[bcal_bool_tr]
tshower_bcal_te[bcal_bool_te] = tShower_te[bcal_bool_te]
tshower_fcal_tr[fcal_bool_tr] = tShower_tr[fcal_bool_tr]
tshower_fcal_te[fcal_bool_te] = tShower_te[fcal_bool_te]

train = train.assign(tShowerBCAL=tshower_bcal_tr)
test = test.assign(tShowerBCAL=tshower_bcal_te)
train = train.assign(tShowerFCAL=tshower_fcal_tr)
test = test.assign(tShowerFCAL=tshower_fcal_te)

### Remove unwated labels
train = train.drop(listvals, axis=1)
test = test.drop(listvals, axis=1)

### Make ptype label intergers instead of strongs
train['ptype'] = train['ptype'].astype(np.int64)

### Make a table of all the Features of interest to sort through overflow values and assign new overflow values

col_names = ['Feature', 'Non-nan Length', 'Median', 'Mode', 'Minimum', 'Maximum', '0.1% Percentile', '99.9% Percentile']
data = []
for i in train.drop('ptype', axis=1).columns:
    try:
        data.append([i, len(train[i][~np.isnan(train[i])]), np.median(train[i][~np.isnan(train[i])]), stats.mode(train[i][~np.isnan(train[i])])[0][0], np.min(train[i][~np.isnan(train[i])]), np.max(train[i][~np.isnan(train[i])]), np.quantile(train[i][~np.isnan(train[i])], 0.001), np.quantile(train[i][~np.isnan(train[i])], 0.999)])
    except:
        data.append([i, len(train[i][~np.isnan(train[i])]), np.median(train[i][~np.isnan(train[i])]), np.nan, np.min(train[i][~np.isnan(train[i])]), np.max(train[i][~np.isnan(train[i])]), np.nan, np.nan])
        
### Fix docaTrack variable

test['docaTrack'][test['docaTrack'] == 10**6] = np.nan
train['docaTrack'][train['docaTrack'] == 10**6] = np.nan

### Seprate datasets by charged or neutral particles

### Charged datasets
test_char = test.loc[np.where(test['q'] != 0)[0]]
train_char = train.loc[np.where(train['q'] != 0)[0]]

### Neutral datasets
test_neut = test.loc[np.where(test['q'] == 0)[0]]
train_neut = train.loc[np.where(train['q'] == 0)[0]]

### Replace all nan values with a set overflow value

labels = np.array(train.columns)[1:]

rep_dict = {'E': -5, 'px':-500, 'py':-500, 'pz':-500, 'q':-10, 'E1E9':-5, 'E9E25': -5, 'docaTrack':-5, 'preshowerE':-5, 'sigLong': -5, 
            'sigTrans':-5, 'sigTheta':-5, 'E_L2':-5, 'E_L3':-5, 'E_L4':-5, 'dEdxCDC':-5, 'dEdxFDC':-5, 'tShower': -10, 'tShowerBCAL': -10, 'tShowerFCAL': -10, 
            'thetac':-5, 'bCalPathLength':-5, 'fCalPathLength':-5, 'dEdxTOF':-5, 'tofTOF':-5, 'pathLengthTOF':-5, 'dEdxSc': -5, 'pathLengthSc':-100, 
            'tofSc':-100, 'xShower': -500, 'yShower':-500, 'zShower':-500, 'xTrack':-500, 'yTrack':-500, 'zTrack':-500, 'CDChits':-5, 
            'FDChits':-5, 'DOCA':-5, 'deltaz':-100, 'deltaphi':-10 , 'tFlightSc':-10, 'tFlightBCAL':-10, 'tFlightTOF':-10, 'tFlightFCAL':-10}


for label in labels:
    train_char[label] = train_char[label].replace(np.nan, rep_dict[label])
    train_neut[label] = train_neut[label].replace(np.nan, rep_dict[label])

    test_char[label] = test_char[label].replace(np.nan, rep_dict[label])
    test_neut[label] = test_neut[label].replace(np.nan, rep_dict[label])

### Save Edited datasets to 'data_processed' folder

new_path = "/Users/erich/Downloads/UConn/Undergraduate-Research/PID_code/data_processed/"

### Save charged datasets
file = data_name + "Training_LE_sorted_charged.hdf5"
filename = new_path + file
train_char.to_hdf(filename, 'event1')

file = data_name + "Test_LE_sorted_charged.hdf5"
filename = new_path + file
test_char.to_hdf(filename, 'event1')


### Save Neutral datsets
file = data_name + "Training_LE_sorted_neutral.hdf5"
filename = new_path + file
train_neut.to_hdf(filename, 'event1')

file = data_name + "Test_LE_sorted_neutral.hdf5"
filename = new_path + file
test_neut.to_hdf(filename, 'event1')