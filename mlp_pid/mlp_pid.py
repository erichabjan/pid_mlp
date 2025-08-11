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


### Suffix for saved data
suffix = '_paper'


### Import models

nn_dirc = '/projects/mccleary_group/habjan.e/PID_code/Main_analysis/NN_models/paper_models/'
charged = tf.keras.models.load_model(nn_dirc + 'Charged_model' + suffix + '.keras')
neutral = tf.keras.models.load_model(nn_dirc + 'Neutral_model' + suffix + '.keras')

### Import Data

dataset_choice = 1
dataset_dic = {1:'pure', 2:'single', 3:'multi'}
data_name = dataset_dic[dataset_choice]

data_path = '/projects/mccleary_group/habjan.e/PID_code/data_processed/'

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

x_neutral = pd.DataFrame.to_numpy(test_neutral.drop(['ptype', 'group', 'true ptype'], axis=1))
y_neutral = np.array(test_neutral['ptype']).astype(np.int64)
group_neutral = np.array(test_neutral['group']).astype(np.int64)
true_neutral = np.array(test_neutral['true ptype']).astype(np.int64)

ptype_dict = {22:0, 130:1, 2112:2, 2212:3, -2212:4, 321:5, -321:6, 11:7, -11:8, 211:9, -211:10, 13:10, -13:9}

y_neutral, true_neutral = np.array([ptype_dict[y_neutral[i]] for i in range(len(y_neutral))]), np.array([ptype_dict[true_neutral[i]] for i in range(len(true_neutral))])
y_charged, true_charged = np.array([ptype_dict[y_charged[i]] for i in range(len(y_charged))]), np.array([ptype_dict[true_charged[i]] for i in range(len(true_charged))])

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

pred_ptype_char = np.argmax(np.maximum.reduceat(pred_char, np.unique(group_charged, return_index=True)[1]), axis=1) + 3. # + 3 to account for neutral particles
pred_ptype_char[max_pred_char < confidence_cut] = 13

### Save data

save_path = '/projects/mccleary_group/habjan.e/PID_code/pid_mlp/paper_plots/'

np.save(save_path + 'charged_mlp_true.npy', true_ptype_char)
np.save(save_path + 'charged_mlp_pred.npy',  pred_ptype_char)

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

import sys
sys.path.append("/projects/mccleary_group/habjan.e/PID_code/pid_mlp/bdt_pid")
from shap_calculation import ShapExplainer
import xgboost as xgb

### Charged MLP

### Number of background per particle
n_background_per_particle = 10**3

### Make mask for background particles

file = data_name + "Training_LE_sorted_charged.hdf5"
filename = data_path + file
train_charged = pd.read_hdf(filename, 'event1')

mask_list = []
part_list = [2212, -2212, 321, -321, 11, -11, 211, -211]
true_charged = np.array(train_charged['ptype'])

for i in part_list:
    
    if i == 211:
        mask_list.append(np.random.choice(np.where((true_charged == i) | (true_charged == -13))[0], n_background_per_particle, replace=False))
    
    elif i == -211:
        mask_list.append(np.random.choice(np.where((true_charged == i) | (true_charged == 13))[0], n_background_per_particle, replace=False))

    else:
        mask_list.append(np.random.choice(np.where(true_charged == i)[0], n_background_per_particle, replace=False))

bg_mask = train_charged.index[np.concatenate(mask_list)]

### Make background array

bg_sample = (
    train_charged.loc[bg_mask]
    .drop(columns=['ptype'], errors='ignore')
    .reset_index(drop=True)
    .to_numpy()
)

charged_explainer = ShapExplainer(charged, test_charged, bg_sample, classes=["p+","K+","mu+pi+","e+","p-","K-","mu-pi-","e-"], n_explain_per_particle = n_background_per_particle)
charged_explainer.calculate_shap()
charged_explainer.save_shap_values("/projects/mccleary_group/habjan.e/PID_code/pid_mlp/paper_plots/", "charged" + suffix)


### Neutral MLP

### Make mask for background particles
file = data_name + "Training_LE_sorted_neutral.hdf5"
filename = data_path + file
train_neutral = pd.read_hdf(filename, 'event1')

mask_list = []
part_list = [22, 2112, 130]
true_neutral = np.array(train_neutral['ptype'])

for i in part_list:
    
    mask_list.append(np.random.choice(np.where(true_neutral == i)[0], n_background_per_particle, replace=False))

bg_mask = train_neutral.index[np.concatenate(mask_list)]

### Make background array

bg_sample = (
    train_neutral.loc[bg_mask]
    .drop(columns=['ptype'], errors='ignore')
    .reset_index(drop=True)
    .to_numpy()
)

neutral_explainer = ShapExplainer(neutral, test_neutral, bg_sample, classes=["gamma","KL","n"], n_explain_per_particle = n_background_per_particle)
neutral_explainer.calculate_shap()
neutral_explainer.save_shap_values("/projects/mccleary_group/habjan.e/PID_code/pid_mlp/paper_plots/", "neutral" + suffix)

print('MLP-PID ran successfully!')