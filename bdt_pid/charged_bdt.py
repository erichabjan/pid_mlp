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

import numpy as np
from sklearn.metrics import accuracy_score
import time


class GroupAccuracyEarlyStopper(TrainingCallback):
    def __init__(self, val_df, val_dmatrix, patience, verbose=True):
        self.val_df = val_df
        self.val_dmatrix = val_dmatrix
        self.patience = patience
        self.verbose = verbose
        self.best_score = -np.inf
        self.best_iteration = 0
        self.wait = 0
        self.ptypes = self.val_df["ptype"].to_numpy()
        self.groups = self.val_df["group"].to_numpy()
        self.true_ptypes = self.val_df["true ptype"].to_numpy()
        self.latest_score = 0
        self.all_groups, self.first_idxs = np.unique(self.groups, return_index=True)
        self.true_labels = self.true_ptypes[self.first_idxs]
    def after_iteration(self, model, epoch, evals_log):
        preds_proba = model.predict(self.val_dmatrix)
        starttime = time.time()

        pred = np.argmax(preds_proba, axis=1)
        conf = np.max(preds_proba, axis=1)

        # Filter: matched predictions only
        matched = (pred == self.ptypes)
        matched_groups = self.groups[matched]
        matched_conf = conf[matched]
        matched_pred = pred[matched]

        # Sort by group, then descending confidence
        sort_idx = np.lexsort((-matched_conf, matched_groups))
        matched_groups = matched_groups[sort_idx]
        matched_pred = matched_pred[sort_idx]

        # Keep best prediction per group
        uniq_groups, first_indices = np.unique(matched_groups, return_index=True)
        best_preds = matched_pred[first_indices]

        # All groups in the full dataset
        

        # Vectorized group index mapping
        result = np.full(len(self.all_groups), 10, dtype=int)
        insert_idx = np.searchsorted(self.all_groups, uniq_groups)
        result[insert_idx] = best_preds
        score = accuracy_score(self.true_labels, result)

        if self.verbose and epoch % 10 == 0:
            print(f"[Custom Metric] Iteration {epoch}: Group Accuracy = {score:.5f}")
            print(f"took {time.time() - starttime:.3f} seconds to evaluate accuracy")
        self.latest_score = score
        if score > self.best_score:
            self.best_score = score
            self.best_iteration = epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Early stopping at iteration {epoch}")
                return True  # stops training

        return False  # continue training


### Import data

dataset_choice = 1
dataset_dic = {1:'pure', 2:'single', 3:'multi'}
data_name = dataset_dic[dataset_choice]

base_path = '/home/rdube/PID_paper/data_processed/'

file = data_name + "Training_LE_sorted_charged.hdf5"
filename = base_path + file
train = pd.read_hdf(filename, 'event1').sample(frac=1)

file = data_name + "Val_LE_sorted_charged.hdf5"
filename = base_path + file
val = pd.read_hdf(filename, 'event1')

## Defines the order of the particles

ptype = {2212:0, 321:1, -13:2, 211:2, -11:3, -2212:4,-321:5,13:6, -211:6,11:7}

## Splitting into x and y
train['ptype'] = train['ptype'].astype(int).map(ptype)
trainx = train.drop(columns=['ptype'])
trainy = train['ptype']
val['true ptype'] = val['true ptype'].astype(int).map(ptype)
val['ptype'] = val['ptype'].astype(int).map(ptype)
valDMatrix = xgb.DMatrix(val.drop(['ptype', 'group', 'true ptype'], axis=1), label=val['true ptype'])
trainDMatrix = xgb.DMatrix(train.drop('ptype', axis=1), label=train['ptype'])

# Make a model for charged particles

def objective(trial):
    callbacks = [GroupAccuracyEarlyStopper(val_df=val, val_dmatrix=valDMatrix, patience=10)]
    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": 8,  # adjust this for your dataset
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    }

    model = xgb.train(params, trainDMatrix, trial.suggest_int("n_estimators", 50, 300), callbacks=callbacks)

    return callbacks[0].latest_score

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
joblib.dump(final_model, "/home/rdube/PID_paper/results/charged_model_bdt.joblib")