import pandas as pd
import xgboost as xgb
from early_stopping_callback import PruneAndEarlyStop
import optuna

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
    callbacks = [PruneAndEarlyStop(val_df=val, val_dmatrix=valDMatrix,match_hypothesis=True, n_ptypes=8, trial=trial)]
    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": 8,  # adjust this for your dataset
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "tree_method": "approx",
        "device":"cuda"
    }

    model = xgb.train(params, trainDMatrix, 500, callbacks=callbacks)

    return callbacks[0].latest_score

# Use Hyperband to prune bad trials early
pruner = optuna.pruners.HyperbandPruner(min_resource=50, max_resource=500, reduction_factor=3)

study = optuna.create_study(direction='maximize',pruner=pruner)
study.optimize(objective, n_trials=100)

# Train final model on full training set
best_params = study.best_trial.params

best_params.update({
    "objective": "multi:softprob",
    "num_class": 8,
    "eval_metric": "mlogloss",
    "use_label_encoder": False,
    "verbosity": 2,
    "tree_method": "approx",
    "device":"cuda"
})

final_model = xgb.train(best_params, trainDMatrix, 500, callbacks=[PruneAndEarlyStop(val_df=val, val_dmatrix=valDMatrix,match_hypothesis=True, n_ptypes=8)])

# Save the model
final_model.save_model("/home/rdube/PID_paper/results/charged_model_bdt.json")