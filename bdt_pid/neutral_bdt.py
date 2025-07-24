import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from early_stopping_callback import PruneAndEarlyStop

from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns

### Import data

dataset_choice = 1
dataset_dic = {1:'pure', 2:'single', 3:'multi'}
data_name = dataset_dic[dataset_choice]

base_path = '/home/rdube/PID_paper/data_processed/'

file = data_name + "Training_LE_sorted_neutral.hdf5"
filename = base_path + file
train = pd.read_hdf(filename, 'event1').sample(frac=1)

file = data_name + "Val_LE_sorted_neutral.hdf5"
filename = base_path + file
val = pd.read_hdf(filename, 'event1')

## Defines the order of the particles

ptype = [22,130,2112]

## Splitting into x and y
train['ptype'] = train['ptype'].astype(int).map(ptype.index)
trainx = train.drop(columns=)
trainy = train['ptype']
val['true ptype'] = val['true ptype'].astype(int).map(ptype.index)
val['ptype'] = val['ptype'].astype(int).map(ptype.index)
valDMatrix = xgb.DMatrix(val.drop(['ptype', 'group', 'true ptype','px', 'py', 'pz', 'q', 'dEdxCDC', 'dEdxFDC', 'thetac', 'bCalPathLength', 'fCalPathLength', 'dEdxTOF', 'tofTOF', 'pathLengthTOF', 'dEdxSc', 'pathLengthSc', 'tofSc', 'xTrack', 'yTrack', 'zTrack', 'CDChits', 'FDChits', 'DOCA', 'deltaz', 'deltaphi'], axis=1), label=val['true ptype'])
trainDMatrix = xgb.DMatrix(trainx, label=trainy)

# Make a model for charged particles

def objective(trial):
    callbacks = [PruneAndEarlyStop(val_df=val, val_dmatrix=valDMatrix,match_hypothesis=False, n_ptypes=3, trial=trial)]
    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": 3, 
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "tree_method": "approx",
        "device":"cuda"
    }

    model = xgb.train(params, trainDMatrix, 500, callbacks=callbacks)
    return callbacks[0].latest_score

# Use Hyperband to prune bad trials early
pruner = optuna.pruners.HyperbandPruner(min_resource=50, max_resource=500, reduction_factor=3)

study = optuna.create_study(direction="maximize", pruner=pruner)
study.optimize(objective, n_trials=100)

# Train final model on full training set
best_params = study.best_trial.params
best_params.update({
    "objective": "multi:softprob",
    "num_class": len(ptype),
    "eval_metric": "mlogloss",
    "verbosity": 2,
    "tree_method": "approx",
    "device":"cuda"
})

final_model = xgb.train(best_params, trainDMatrix, 500, callbacks=[PruneAndEarlyStop(val_df=val, val_dmatrix=valDMatrix,match_hypothesis=False, n_ptypes=3)])

# Save the model
final_model.save_model("/home/rdube/PID_paper/results/neutral_model_bdt.json")