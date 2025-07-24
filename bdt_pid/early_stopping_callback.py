import numpy as np
import optuna
from xgboost.callback import TrainingCallback
import time
import pandas as pd
from sklearn.metrics import accuracy_score

class PruneAndEarlyStop(TrainingCallback):
    def __init__(self, val_df, val_dmatrix, match_hypothesis=True, n_ptypes = 8, patience=5, trial=None, verbose=True):
        self.val_df = val_df
        self.trial = trial
        self.val_dmatrix = val_dmatrix
        self.patience = patience
        self.verbose = verbose
        self.best_score = -np.inf
        self.best_iteration = 0
        self.wait = 0
        self.n_ptypes = n_ptypes
        self.match_hypothesis = match_hypothesis
        self.latest_score = 0
        self.ptypes = None
        self.groups = None
        self.true_ptypes = None
        self.all_groups = None
        self.true_labels = None
        self.first_idxs = None
        self.true_ptypes = self.val_df["true ptype"].to_numpy()
        self.true_labels = self.true_ptypes
        if self.match_hypothesis:
            self.ptypes = self.val_df["ptype"].to_numpy()
            self.groups = self.val_df["group"].to_numpy()
            self.all_groups, self.first_idxs = np.unique(self.groups, return_index=True)
            self.true_labels = self.true_ptypes[self.first_idxs]
            
    def after_iteration(self, model, epoch, evals_log):
        preds_proba = model.predict(self.val_dmatrix)
        pred = np.argmax(preds_proba, axis=1)
        conf = np.max(preds_proba, axis=1)
        starttime = time.time()
        if self.match_hypothesis:
            # Filter: matched predictions only
            matched = (pred == self.ptypes) & (conf>0.4)
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
            # Vectorized group index mapping
            result = np.full(len(self.all_groups), self.n_ptypes+1, dtype=int)
            insert_idx = np.searchsorted(self.all_groups, uniq_groups)
            result[insert_idx] = best_preds
        else:
            result = pred
            result[conf < 0.4] = self.n_ptypes + 1
        
        
        score = accuracy_score(self.true_labels, result)

        if self.verbose and epoch % 10 == 0:
            print(f"[Custom Metric] Iteration {epoch}: Group Accuracy = {score:.5f}")
            print(f"took {time.time() - starttime:.3f} seconds to evaluate accuracy")
        self.latest_score = score

        if self.trial is not None:
            self.trial.report(score, step=epoch)
            if self.trial.should_prune():
                print(f"[Optuna] Trial pruned at epoch {epoch} with Accuracy {score:.5f}")
                raise optuna.exceptions.TrialPruned()
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