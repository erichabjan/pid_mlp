import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import time
import xgboost as xgb
from xgboost.callback import TrainingCallback
import optuna

from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns
import joblib

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
trainx = train.drop(columns=['ptype'])
trainy = train['ptype']
val['true ptype'] = val['true ptype'].astype(int).map(ptype.index)
val['ptype'] = val['ptype'].astype(int).map(ptype.index)
valDMatrix = xgb.DMatrix(val.drop(['ptype', 'group', 'true ptype'], axis=1), label=val['true ptype'])
trainDMatrix = xgb.DMatrix(train.drop('ptype', axis=1), label=train['ptype'])

# Make a model for charged particles

def objective(trial):
    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": 10,  # adjust this for your dataset
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    }

    model = xgb.train(params, trainDMatrix, trial.suggest_int("n_estimators", 50, 300), evals=[(trainDMatrix,"train"),(valDMatrix, 'val')], early_stopping_rounds=10)

    preds = np.argmax(model.predict(valDMatrix),axis=1)
    return accuracy_score(val['true ptype'],preds)

# Use Hyperband to prune bad trials early
pruner = optuna.pruners.HyperbandPruner(min_resource=50, max_resource=300, reduction_factor=3)

study = optuna.create_study(direction="maximize", pruner=pruner)
study.optimize(objective, n_trials=50)

# Train final model on full training set
best_params = study.best_trial.params
best_params.update({
    "objective": "multi:softprob",
    "num_class": len(ptype),
    "eval_metric": "mlogloss",
    "use_label_encoder": False,
    "verbosity": 2
})

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(trainx, trainy)

# Save the model
joblib.dump(final_model, "/home/rdube/PID_paper/results/neutral_model_bdt.joblib")