import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from sklearn.metrics import accuracy_score, confusion_matrix

import os
import pickle
import shap
from matplotlib.ticker import LogLocator, NullLocator
import xgboost as xgb

from shap_calculation import ShapExplainer

neutral_ptype = [22,130,2112]
neutral = xgb.Booster()
neutral.load_model('/home/rdube/PID_paper/results/neutral_model_bdt.json')

test_neutral = pd.read_hdf('/home/rdube/PID_paper/data_processed/pureTest_LE_sorted_neutral.hdf5', 'event1')
test_neutral['ptype'] = test_neutral['ptype'].astype(int).map(neutral_ptype.index)
test_neutral['group'] = test_neutral['group'].astype(int)
test_neutral['true ptype']= test_neutral['true ptype'].astype(int).map(neutral_ptype.index)
x_neutral = test_neutral.drop(columns=['px', 'py', 'pz', 'q', 'dEdxCDC', 'dEdxFDC', 'thetac', 'bCalPathLength', 'fCalPathLength', 'dEdxTOF', 'tofTOF', 'pathLengthTOF', 'dEdxSc', 'pathLengthSc', 'tofSc', 'xTrack', 'yTrack', 'zTrack', 'CDChits', 'FDChits', 'DOCA', 'deltaz', 'deltaphi'])
x_neutral_DMatrix = xgb.DMatrix(x_neutral.drop(columns=['ptype','true ptype','group']))
y_neutral = test_neutral['true ptype'].to_numpy()
group_neutral = test_neutral['group'].to_numpy()
neutral.set_param({"device":"cuda"})
pred_neut = neutral.predict(x_neutral_DMatrix)

neutral_explainer = ShapExplainer(neutral, x_neutral, None, classes=["gamma","KL","n"])
neutral_explainer.calculate_shap()
neutral_explainer.plot_summary(10)

### Charged Particle PID

## Define the order of the particle labels
charged_ptype = {2212:0, 321:1, -13:2, 211:2, -11:3, -2212:4,-321:5,13:6, -211:6,11:7}

## Import model

charged = xgb.Booster()
charged.load_model('/home/rdube/PID_paper/results/charged_model_bdt.json')
## Import Data

test_charged = pd.read_hdf('/home/rdube/PID_paper/data_processed/pureTest_LE_sorted_charged.hdf5', 'event1')
train_charged = pd.read_hdf('/home/rdube/PID_paper/data_processed/pureTraining_LE_sorted_charged.hdf5', 'event1')

test_charged['ptype'] = test_charged['ptype'].astype(int).map(charged_ptype)
test_charged['group'] = test_charged['group'].astype(int)
test_charged['true ptype']= test_charged['true ptype'].astype(int).map(charged_ptype)

train_charged['ptype'] = train_charged['ptype'].astype(int).map(charged_ptype)
x_charged = test_charged.drop(['ptype', 'group', 'true ptype'], axis=1)
x_charged_DMatrix = xgb.DMatrix(x_charged)
y_charged = np.array(test_charged['ptype'].astype(np.int64).map(charged_ptype))
group_charged = np.array(test_charged['group'].astype(np.int64))
true_charged = np.array(test_charged['true ptype'].astype(np.int64).map(charged_ptype))


### Make particle identification predictions

pred_char = charged.predict(x_charged_DMatrix)

confidence_cut = 0.4

pred_ind_char = np.argmax(pred_char, axis=1)
conf = np.max(pred_char, axis=1)
match = (conf>confidence_cut) & (pred_ind_char == y_charged)

matched_idx = np.where(match)[0]
matched_groups = group_charged[matched_idx]
matched_conf = conf[matched_idx]
matched_pred = y_charged[matched_idx]

sort_order = np.lexsort((-matched_conf, matched_groups))  # sort by group, then -conf
sorted_groups = matched_groups[sort_order]
sorted_preds = matched_pred[sort_order]

_, first_indices = np.unique(sorted_groups, return_index=True)
best_groups = sorted_groups[first_indices]
best_preds = sorted_preds[first_indices]

group_ids, first_idxs = np.unique(group_charged, return_index=True)
pred_ptype_char = np.full(len(group_ids), 9, dtype=int)
group_to_index = {g: i for i, g in enumerate(group_ids)}
for g, p in zip(best_groups, best_preds):
    pred_ptype_char[group_to_index[g]] = p

true_ptype_char = true_charged[first_idxs]

save_path = '/home/rdube/PID_paper/pid_mlp/bdt_plots/'

np.save(save_path + 'charged_bdt_true.npy', true_ptype_char)
np.save(save_path + 'charged_bdt_pred.npy',  pred_ptype_char)


### pick the hypothesis with the highest confidence for neutral particles

### Classify particles for each event using highest confidence
groups, true_group_ind = np.unique(group_neutral, return_index=True)
true_ptype_neut = y_neutral[true_group_ind]

pred_neut_event = np.maximum.reduceat(pred_neut, np.unique(group_neutral, return_index=True)[1])
pred_ind_neut = np.argmax(np.maximum.reduceat(pred_neut, np.unique(group_neutral, return_index=True)[1]), axis=1)
max_pred_neut = pred_neut_event[np.arange(len(pred_ind_neut)), pred_ind_neut]

pred_ptype_neut = np.argmax(np.maximum.reduceat(pred_neut, np.unique(group_neutral, return_index=True)[1]), axis=1) 
pred_ptype_neut[max_pred_neut < confidence_cut] = 3

np.save(save_path + 'neutral_bdt_true.npy', true_ptype_neut)
np.save(save_path + 'neutral_bdt_pred.npy',  pred_ptype_neut)

charged_explainer = ShapExplainer(charged, test_charged, train_charged, classes=["p+","K+","mu+pi+","e+","p-","K-","mu-pi-","e-"])
charged_explainer.calculate_shap()
charged_explainer.plot_summary(10)

