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
from misc import *

from shap_calculation import ShapExplainer
from misc import load_file, make_predictions

save_path = '/home/rdube/PID_paper/pid_mlp/bdt_plots/'

neutral_ptype = [22,130,2112]
neutral = xgb.Booster()
neutral.load_model('/home/rdube/PID_paper/results/neutral_model_bdt.json')
neutral.set_param({"device":"cuda"})

test_neutral = load_file("neutral","Test", neutral_ptype.index)
pred_ptype_neut, true_ptype_neut = make_predictions(neutral, test_neutral, match_hypotheses=False)


np.save(save_path + 'neutral_bdt_true.npy', true_ptype_neut)
np.save(save_path + 'neutral_bdt_pred.npy',  pred_ptype_neut)

neutral_explainer = ShapExplainer(neutral, test_neutral, None, classes=["gamma","KL","n"], n_explain_per_particle=1000)
neutral_explainer.calculate_shap()
neutral_explainer.save_shap_values("/home/rdube/PID_paper/pid_mlp/bdt_plots/","neutral")

### Charged Particle PID

## Define the order of the particle labels
charged_ptype = {2212:0, 321:1, -13:2, 211:2, -11:3, -2212:4,-321:5,13:6, -211:6,11:7}

## Import model

charged = xgb.Booster()
charged.load_model('/home/rdube/PID_paper/results/charged_model_bdt.json')
charged.set_param({"device":"cuda"})
## Import Data

test_charged = load_file("charged","Test", charged_ptype)


### Make particle identification predictions

pred_ptype_char, true_ptype_char = make_predictions(charged, test_charged)

np.save(save_path + 'charged_bdt_true.npy', true_ptype_char)
np.save(save_path + 'charged_bdt_pred.npy',  pred_ptype_char)

### pick the hypothesis with the highest confidence for neutral particles

charged_explainer = ShapExplainer(charged, test_charged, None, classes=["p+","K+","mu+pi+","e+","p-","K-","mu-pi-","e-"], n_explain_per_particle=1000)
charged_explainer.calculate_shap()
charged_explainer.save_shap_values("/home/rdube/PID_paper/pid_mlp/bdt_plots/","charged")


### Making SHAP Plots
charged_indices = np.load("/home/rdube/PID_paper/pid_mlp/bdt_plots/charged_indices_to_test.npy")
charged_data_to_test = test_charged.loc[charged_indices].drop(columns=['group','ptype','true ptype'])
charged_shaps = np.load("/home/rdube/PID_paper/pid_mlp/bdt_plots/charged_shap.npy")
SHAP_plots(charged_shaps, charged_data_to_test, charged_shaps, charged_data_to_test, "pos")
SHAP_plots(charged_shaps, charged_data_to_test, charged_shaps, charged_data_to_test, "neg")


neutral_indices = np.load("/home/rdube/PID_paper/pid_mlp/bdt_plots/neutral_indices_to_test.npy")
neutral_data_to_test = test_neutral.loc[neutral_indices].drop(columns=['group','ptype','true ptype'])
neutral_shaps = np.load("/home/rdube/PID_paper/pid_mlp/bdt_plots/neutral_shap.npy")
SHAP_plots(neutral_shaps, neutral_data_to_test, neutral_shaps, neutral_data_to_test, "neutral")