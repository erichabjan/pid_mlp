import pandas as pd
import xgboost as xgb
from early_stopping_callback import PruneAndEarlyStop
import optuna
from misc import load_file

ptype = {2212:0, 321:1, -13:2, 211:2, -11:3, -2212:4,-321:5,13:6, -211:6,11:7}
### Import data

train = load_file("charged", "Training",ptype)
val = load_file("charged","Val",ptype)

## Defines the order of the particles



## Splitting into x and y
valDMatrix = xgb.DMatrix(val.drop(['ptype', 'group', 'true ptype'], axis=1), label=val['true ptype'], missing = float("NaN"))
trainDMatrix = xgb.DMatrix(train.drop('ptype', axis=1), label=train['ptype'], missing = float("NaN"))

# Make a model for charged particles

def objective(trial):
    callbacks = [PruneAndEarlyStop(val_df=val, val_dmatrix=valDMatrix,match_hypothesis=True, n_ptypes=8, trial=trial)]
    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": 8,  
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