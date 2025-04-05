import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

import tensorflow as tf
import tensorflow_datasets as tfds
import keras_tuner as kt
import keras.backend as K

from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns

import os
import pickle
import shap
from matplotlib.ticker import LogLocator, NullLocator


### Import models

charged = tf.keras.models.load_model('/Users/erich/Downloads/UConn/Undergraduate-Research/PID_code/Main_analysis/NN_models/Charged_model_1hidden.keras') #_1hidden.keras')
neutral = tf.keras.models.load_model('/Users/erich/Downloads/UConn/Undergraduate-Research/PID_code/Main_analysis/NN_models/Neutral_model_best.keras') #_1hidden.keras')

### Import Data

dataset_choice = 1
dataset_dic = {1:'pure', 2:'single', 3:'multi'}
data_name = dataset_dic[dataset_choice]

#data_path = os.getcwd().replace('Main_analysis', 'data_processed/')
data_path = '/Users/erich/Downloads/UConn/Undergraduate-Research/PID_code/data_processed/'

file = data_name + "Test_LE_sorted_charged.hdf5"
filename = data_path + file
test_charged = pd.read_hdf(filename, 'event1')

file = data_name + "Test_LE_sorted_neutral.hdf5"
filename = data_path + file
test_neutral = pd.read_hdf(filename, 'event1')

x_charged = pd.DataFrame.to_numpy(test_charged.drop(['ptype', 'group', 'true ptype'], axis=1))
y_charged = np.array(test_charged['ptype']).astype(np.int64)
group_charged = np.array(test_charged['group']).astype(np.int64)
true_charged = np.array(test_charged['true ptype']).astype(np.int64)

x_charged = x_charged[:, :38]

x_neutral = pd.DataFrame.to_numpy(test_neutral.drop(['ptype', 'group', 'true ptype'], axis=1))
y_neutral = np.array(test_neutral['ptype']).astype(np.int64)
group_neutral = np.array(test_neutral['group']).astype(np.int64)
true_neutral = np.array(test_neutral['true ptype']).astype(np.int64)

x_neutral = x_neutral[:, :38]

ptype_dict = {22:0, 130:1, 2112:2, 2212:3, -2212:4, 321:5, -321:6, 11:7, -11:8, 211:9, -211:10, 13:11, -13:12}

y_neutral, true_neutral = np.array([ptype_dict[y_neutral[i]] for i in range(len(y_neutral))]), np.array([ptype_dict[true_neutral[i]] for i in range(len(true_neutral))])
y_charged, true_charged = np.array([ptype_dict[y_charged[i]] for i in range(len(y_charged))]), np.array([ptype_dict[true_charged[i]] for i in range(len(true_charged))])

### Make lists of test data

labels = np.array(['E', 'px', 'py', 'pz', 'q', 'E1E9', 'E9E25', 'docaTrack',
       'preshowerE', 'sigLong', 'sigTrans', 'sigTheta', 'E_L2', 'E_L3', 'E_L4',
       'dEdxCDC', 'dEdxFDC', 'tShower', 'thetac', 'bCalPathLength',
       'fCalPathLength', 'dEdxTOF', 'tofTOF', 'pathLengthTOF', 'dEdxSc',
       'pathLengthSc', 'tofSc', 'xShower', 'yShower', 'zShower', 'xTrack',
       'yTrack', 'zTrack', 'CDChits', 'FDChits', 'DOCA', 'deltaz', 'deltaphi'])

x_charged = np.transpose(np.array([np.array(test_charged[labels[j]]) for j in range(len(labels))]))
x_neutral = np.transpose(np.array([np.array(test_neutral[labels[j]]) for j in range(len(labels))]))

### Make particle identification predictions

pred_neut = neutral.predict(x_neutral)
pred_char = charged.predict(x_charged)

# Prediction-based PID 

### pick the hypothesis with the highest confidence for charged particles

confidence_cut = 0.4
### Classify particles for each event using highest confidence
groups, true_group_ind = np.unique(group_charged, return_index=True)
true_ptype_char = true_charged[true_group_ind]

pred_char_event = np.maximum.reduceat(pred_char, np.unique(group_charged, return_index=True)[1])
pred_ind_char = np.argmax(np.maximum.reduceat(pred_char, np.unique(group_charged, return_index=True)[1]), axis=1)
max_pred_char = pred_char_event[np.arange(len(pred_ind_char)), pred_ind_char]

pred_ptype_char = np.argmax(np.maximum.reduceat(pred_char, np.unique(group_charged, return_index=True)[1]), axis=1) + 3
pred_ptype_char[max_pred_char < confidence_cut] = 13

### Combine muons and pions

true_plot = np.copy(true_ptype_char)
true_plot[true_plot == 12] = 9
true_plot[true_plot == 11] = 10

PID_plot = np.copy(pred_ptype_char)
PID_plot[PID_plot == 12] = 9
PID_plot[PID_plot == 11] = 10

save_path = '/Users/erich/Downloads/UConn/Undergraduate-Research/PID_code/pid_mpl/paper_plots/'

np.save(save_path + 'charged_mlp_true.npy', true_plot)
np.save(save_path + 'charged_mlp_pred.npy',  PID_plot)

### pick the hypothesis with the highest confidence for neutral particles

### Classify particles for each event using highest confidence
groups, true_group_ind = np.unique(group_neutral, return_index=True)
true_ptype_neut = true_neutral[true_group_ind]

pred_neut_event = np.maximum.reduceat(pred_neut, np.unique(group_neutral, return_index=True)[1])
pred_ind_neut = np.argmax(np.maximum.reduceat(pred_neut, np.unique(group_neutral, return_index=True)[1]), axis=1)
max_pred_neut = pred_neut_event[np.arange(len(pred_ind_neut)), pred_ind_neut]

pred_ptype_neut = np.argmax(np.maximum.reduceat(pred_neut, np.unique(group_neutral, return_index=True)[1]), axis=1) 
pred_ptype_neut[max_pred_neut < confidence_cut] = 13

np.save(save_path + 'neutral_mlp_true.npy', true_ptype_neut)
np.save(save_path + 'neutral_mlp_pred.npy',  pred_ptype_neut)

# Shapley Values

### Use SHAP to interpret the most valuable features in each dataset for both models

samplesize = 10**3
shap_values_neut = []
shap_values_char = []

for i in range(13):
    if i == 0 or i == 1 or i == 2:
        bacvals_neut = np.random.choice(np.where(true_neutral == i)[0], samplesize, replace=False)
        background_neut = x_neutral[bacvals_neut]
        e_neut = shap.DeepExplainer(neutral, background_neut);
        shap_values_neut_out = e_neut.shap_values(background_neut);
        shap_values_neut.append(np.array(shap_values_neut_out)[:, :, i])

    else: 
        bacvals_char = np.random.choice(np.where(true_charged == i)[0], samplesize, replace=False)
        background_char = x_charged[bacvals_char]
        e_char = shap.DeepExplainer(charged, background_char);
        shap_values_char_out = e_char.shap_values(background_char);
        shap_values_char.append(np.array(shap_values_char_out)[:, :, i-3])
        
### average the shap value for each particle

pshaplist = []

ptypedic = {0:r'$\gamma$', 1:r'$K_{L}^{0}$', 2:r'$n$', 3:r'$p$', 4:r'$\bar{p}$', 
            5:r'$K^{+}$', 6:r'$K^{-}$', 7:r'$e^{-}$', 8:r'$e^{+}$', 9:r'$\pi^{+}$', 
            10:r'$\pi^{-}$', 11:r'$\mu^{-}$', 12:r'$\mu^{+}$'}

shap_values_neut_arr = np.array(shap_values_neut)
shap_values_char_arr = np.array(shap_values_char)

### Take the median value for each feature for each neutral particle
for j in range(3): 
    pshaplist.append(np.nanmedian(shap_values_neut_arr[j, :, :], axis=0))


### Take the median value for each feature for each charged particle
for j in range(10):
    pshaplist.append(np.nanmedian(shap_values_char_arr[j, :, :], axis=0))
    
    

dcols = np.array(['E', 'px', 'py', 'pz', 'q', 'E1E9', 'E9E25', 'docaTrack',
       'preshowerE', 'sigLong', 'sigTrans', 'sigTheta', 'E_L2', 'E_L3', 'E_L4',
       'dEdxCDC', 'dEdxFDC', 'tShower', 'thetac', 'bCalPathLength',
       'fCalPathLength', 'dEdxTOF', 'tofTOF', 'pathLengthTOF', 'dEdxSc',
       'pathLengthSc', 'tofSc', 'xShower', 'yShower', 'zShower', 'xTrack',
       'yTrack', 'zTrack', 'CDChits', 'FDChits', 'DOCA', 'deltaz', 'deltaphi'])
    
### Turn shapley values into a pandas dataframe

shap_dict_gamma = {}
shap_dict_klong = {}
shap_dict_neutron = {}

shap_dict_proton = {}
shap_dict_pbar = {}
shap_dict_kplus = {}
shap_dict_kminus = {}

shap_dict_electron = {}
shap_dict_positron = {}
shap_dict_piplus = {}
shap_dict_piminus = {}

shap_list = [shap_dict_gamma, shap_dict_klong, shap_dict_neutron, 
             shap_dict_proton, shap_dict_pbar, shap_dict_kplus, shap_dict_kminus, 
             shap_dict_electron, shap_dict_positron, shap_dict_piplus, shap_dict_piminus]

df_list = []

for i in range(len(ptype_dict) - 2):

    if i == 0 or i == 1 or i == 2:
        for j in range(len(dcols)):
            if any(np.concatenate(abs(np.array(shap_values_neut)[:, :, j])) != 0):
                shap_list[i][dcols[j]] = abs(shap_values_neut[i][:, j])
        df_list.append(pd.DataFrame(shap_list[i]))

    elif i == 3 or i == 4 or i == 5 or i == 6 or i == 7  or i == 8: 
        for j in range(len(dcols)):
            if any(np.concatenate(abs(np.array(shap_values_char)[:, :, j])) != 0):
                shap_list[i - 3][dcols[j]] = abs(shap_values_char[i- 3][:, j])
        df_list.append(pd.DataFrame(shap_list[i - 3]))

    elif i == 9: 
        for j in range(len(dcols)):
            if any(np.concatenate(abs(np.array(shap_values_char)[:, :, j])) != 0):
                shap_list[i - 3][dcols[j]] = abs(shap_values_char[i- 3][:, j])
        piplus_df = pd.DataFrame(shap_list[i - 3])

        muplus_df = {}
        for j in range(len(dcols)):
            if any(np.concatenate(abs(np.array(shap_values_char)[:, :, j])) != 0):
                muplus_df[dcols[j]] = abs(shap_values_char[12 - 3][:, j])
        muplus_df = pd.DataFrame(muplus_df)
        
        df_list.append(pd.concat([piplus_df, muplus_df], ignore_index=True))

    elif i == 10: 
        for j in range(len(dcols)):
            if any(np.concatenate(abs(np.array(shap_values_char)[:, :, j])) != 0):
                shap_list[i - 3][dcols[j]] = abs(shap_values_char[i- 3][:, j])
        piminus_df = pd.DataFrame(shap_list[i - 3])

        muminus_df = {}
        for j in range(len(dcols)):
            if any(np.concatenate(abs(np.array(shap_values_char)[:, :, j])) != 0):
                muminus_df[dcols[j]] = abs(shap_values_char[11 - 3][:, j])
        muminus_df = pd.DataFrame(muminus_df)
        
        df_list.append(pd.concat([piminus_df, muminus_df], ignore_index=True))


with open(save_path + 'shapley_values_df.pkl', 'wb') as f:
    pickle.dump(df_list, f)