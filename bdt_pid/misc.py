import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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
    data = xgb.DMatrix(data_raw.drop(columns=['ptype','true ptype','group']), missing=float("NaN"))
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

def feature_order(bdt_shap_vals, mlp_shap_vals, nfeatures=5):
    mean_bdt = np.average(np.abs(bdt_shap_vals), axis=0)
    mean_mlp = np.average(np.abs(mlp_shap_vals), axis=0)
    feature_order = np.argsort((mean_bdt+mean_mlp)/2, axis=0)
    feature_order = feature_order[-min(nfeatures, len(feature_order)) :]
    return feature_order

# === MODIFIED VIOLIN FUNCTION START ===
# This is a modified version of the violin plot from the SHAP library, with added flexibility for using axes and simplified to use the formatting in this paper
def violin(shap_values, ax, features=None,max_display=5,color="coolwarm",layered_violin_max_num_bins=10,x_label="SHAP Values",show_label_names=True, feature_order=[]):
    feature_names = features.columns
    features = features.to_numpy()
    ax.set_xscale("symlog")
    ax.axvline(x=0, color="#999999", zorder=-1)
    
    num_x_points = 200
    # Use nanquantile to be safe from NaNs in SHAP values
    shap_min, shap_max = np.nanquantile(shap_values, [0.001, 0.999]) * 1.5
    if np.isnan(shap_min): shap_min = -1
    if np.isnan(shap_max): shap_max = 1
    x_points = np.linspace(shap_min, shap_max, num_x_points)
    
    # loop through each feature and plot:
    for pos, ind in enumerate(feature_order):
        # 1. DATA SEPARATION
        feature_all = features[:, ind]
        shap_values_all = shap_values[:, ind]
        nan_mask = np.isnan(feature_all)
        
        nan_shaps = shap_values_all[nan_mask]
        
        feature_non_nan = feature_all[~nan_mask]
        shap_values_non_nan = shap_values_all[~nan_mask]
        
        # --- KDE Calculation ---
        all_ys = []
        colors = []
            
        # 2. Handle non-NaN data (colored layers)
        if len(shap_values_non_nan) > 1:
            # Decide binning strategy for non-NaN data
            unique, counts = np.unique(feature_non_nan, return_counts=True)
            if unique.shape[0] > 1 and unique.shape[0] <= layered_violin_max_num_bins:
                order = np.argsort(unique)
                thesebins = np.cumsum(counts[order])
                thesebins = np.insert(thesebins, 0, 0)
            else:
                thesebins = (np.linspace(0, feature_non_nan.shape[0], layered_violin_max_num_bins + 1).round(0).astype("int"))
            
            nbins = thesebins.shape[0] - 1
            if nbins < 1: nbins = 1

            # Order the non-NaN feature data so we can apply percentiling
            order = np.argsort(feature_non_nan)
            
            cmap = plt.get_cmap(color)
            
            # Calculate KDE for each bin
            for i in range(nbins):
                shaps_in_bin = shap_values_non_nan[order[thesebins[i] : thesebins[i + 1]]]
                if shaps_in_bin.shape[0] < 2:
                    continue # Skip bins with less than 2 points for KDE
                
                kde = gaussian_kde(shaps_in_bin + np.random.normal(loc=0, scale=0.001, size=shaps_in_bin.shape[0]))(x_points)
                kde *= len(shaps_in_bin) # Scale KDE by number of points in the bin
                
                all_ys.append(kde)
                colors.append(cmap(i / (nbins - 1) if nbins > 1 else 0.5))

        # 3. Handle NaN data  (will be the last layer, colored gray)
        if len(nan_shaps) > 1:
            nan_kde = gaussian_kde(nan_shaps + np.random.normal(loc=0, scale=0.001, size=nan_shaps.shape[0]))(x_points)
            nan_kde *= len(nan_shaps) # Scale by number of points
            all_ys.append(nan_kde)
            colors.append(mcolors.to_rgba('gray')) # Assign gray color for NaNs
        # 4. PLOTTING
        if not all_ys: # If no data to plot for this feature, skip
            continue

        all_ys = np.array(all_ys)
        all_ys_cumulative = np.cumsum(all_ys, axis=0)
        
        # Define violin plot width and find scale factor
        width = 0.8
        if np.max(all_ys_cumulative) > 0:
            scale = np.max(all_ys_cumulative) * 2 / width
        else:
            scale = 1
        
        # Plot each layer, from the outside in (i.e., largest cumulative area first)
        for i in range(len(all_ys) - 1, -1, -1):
            y = all_ys_cumulative[i, :] / scale
            c = colors[i]
            ax.fill_between(x_points, pos - y, pos + y, facecolor=c, edgecolor="face")
    ax.set_xlim(shap_min, shap_max)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(color="#333333", labelcolor="#333333")
    if show_label_names:
        ax.set_yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=10)
    else:
        ax.set_yticks([])
    ax.tick_params("y", length=5, width=0.5, which="major",pad=1)
    ax.tick_params("x", labelsize=11,pad=1)
    ax.set_ylim(-1, len(feature_order))
    ax.set_xlabel(x_label, fontsize=13)

def SHAP_plots(mlp_shap_vals, mlp_data, bdt_shap_vals, bdt_data, charge):
    if charge == "pos":
        nClasses = 4
        bdt_indices = [0,1,2,3]
        mlp_indices = [0,2,6,5]
        classes = [r'$p$',r'$K^{+}$',r'$\pi^{+}$ | $\mu^{+}$', r'$e^{+}$']
    elif charge=="neg":
        nClasses = 4
        bdt_indices = [4,5,6,7]
        mlp_indices = [1,3,7,4]
        classes = [r'$\bar{p}$', r'$K^{-}$', r'$\pi^{-}$ | $\mu^{-}$', r'$e^{-}$']
    else:
        nClasses = 3
        bdt_indices = [0,1,2]
        mlp_indices = [0,1,2]
        classes = [r'$\gamma$', r'$K_{L}^{0}$', r'$n$']
    fig, axes = plt.subplots(nrows=nClasses, ncols=2, figsize=(10,11))
    for i in range(nClasses):
        if i == nClasses-1:
            BDT_x_label = "BDT SHAP Values"
            MLP_x_label = "MLP SHAP Values"
        else:
            BDT_x_label = ""
            MLP_x_label = ""
        fo = feature_order(bdt_shap_vals[bdt_indices[i]],mlp_shap_vals[mlp_indices[i]])
        violin(mlp_shap_vals[mlp_indices[i]],axes[i][0],features=mlp_data,x_label=MLP_x_label, feature_order=fo)
        violin(bdt_shap_vals[bdt_indices[i]],axes[i][1],features=bdt_data,x_label=BDT_x_label, show_label_names=False, feature_order=fo)
    #plt.tight_layout()
    fig.subplots_adjust(left=0.12,right=0.98,wspace=0.1,hspace=0.3, bottom=0, top=0.95)
    m = cm.ScalarMappable(cmap=plt.get_cmap("coolwarm"))
    m.set_array([0,1])
    cb = fig.colorbar(m, ax=axes, ticks=[0,1],location='bottom',aspect=80,fraction=0.05,pad=0.06)
    cb.set_ticklabels(["Low", "High"])
    cb.set_label("Feature Values", size=12, labelpad=0)
    cb.ax.tick_params(axis='y', labelsize=11, length=0)
    cb.ax.yaxis.set_ticks_position('both')
    cb.set_alpha(1)
    cb.outline.set_visible(False)

    #Find the position of the text relative to the left column axes
    belowBox = axes[1][0].get_position()
    aboveBox = axes[0][0].get_position()
    leftBox = axes[0][0].get_position()
    rightBox = axes[0][1].get_position()

    rel_pos_x = (rightBox.x0 - leftBox.x1)/2
    rel_pos_y = (aboveBox.y0 - belowBox.y1)/2 *1/5
    for j in range(nClasses-1,-1,-1):
        leftBox = axes[j][0].get_position()
        fig.text(leftBox.x1+rel_pos_x, leftBox.y1+rel_pos_y, "SHAP Values For: " + classes[j], ha='center', va='center', fontsize=13)
    fig.savefig("/home/rdube/PID_paper/pid_mlp/bdt_plots/" + charge + "_SHAP.png",dpi=300)


