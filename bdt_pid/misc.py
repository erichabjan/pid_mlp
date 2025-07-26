import pandas as pd
import xgboost as xgb
import numpy as np
def load_file(charge,dataset, map):
    fin = '/home/rdube/PID_paper/data_processed/pure' + dataset +'_LE_sorted_'+ charge + '.hdf5'
    loaded_file = pd.read_hdf(fin, 'event1')
    if 'ptype' in loaded_file.columns:
        loaded_file['ptype'] = loaded_file['ptype'].astype(int).map(map)
    if 'true ptype' in loaded_file.columns:
        loaded_file['true ptype'] = loaded_file['true ptype'].astype(int).map(map)
    if 'group' in loaded_file.columns:
        loaded_file['group'] = loaded_file['group'].astype(int)
    return loaded_file

def make_predictions(model, data_raw,match_hypotheses=True, confidence_cut = 0.4):

    hypotheses = np.array(data_raw['ptype'])
    groups = np.array(data_raw['group'])
    true_ptypes = np.array(data_raw['true ptype'])
    data = xgb.DMatrix(data_raw.drop(columns=['ptype','true ptype','group']))
    preds = model.predict(data)
    if match_hypotheses:
        
        pred_inds = np.argmax(preds, axis=1)
        conf = np.max(preds, axis=1)
        match = (conf>confidence_cut) & (pred_inds == hypotheses)

        matched_idx = np.where(match)[0]
        matched_groups = groups[matched_idx]
        matched_conf = conf[matched_idx]
        matched_pred = hypotheses[matched_idx]

        sort_order = np.lexsort((-matched_conf, matched_groups))  # sort by group, then -conf
        sorted_groups = matched_groups[sort_order]
        sorted_preds = matched_pred[sort_order]

        _, first_indices = np.unique(sorted_groups, return_index=True)
        best_groups = sorted_groups[first_indices]
        best_preds = sorted_preds[first_indices]

        group_ids, first_idxs = np.unique(groups, return_index=True)
        preds_final = np.full(len(group_ids), 9, dtype=int)
        group_to_index = {g: i for i, g in enumerate(group_ids)}
        for g, p in zip(best_groups, best_preds):
            preds_final[group_to_index[g]] = p

        true_ptypes_final = true_ptypes[first_idxs]
    else:
        preds_final = np.argmax(preds, axis=1)
        true_ptypes_final = true_ptypes
    return preds_final, true_ptypes_final