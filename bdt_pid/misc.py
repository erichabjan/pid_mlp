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

## This is a modified version of the violin plot from the SHAP library, with added flexibility for using axes and simplified to use the formatting in this paper
def violin(shap_values, ax, features=None,max_display=5,color="coolwarm",layered_violin_max_num_bins=20,x_label="SHAP Values",show_label_names=True):
    feature_names = features.columns
    features = features.to_numpy()
    ax.set_xscale("symlog")
    feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
    feature_order = feature_order[-min(max_display, len(feature_order)) :]
    ax.axvline(x=0, color="#999999", zorder=-1)
    num_x_points = 200
    bins = (np.linspace(0, features.shape[0], layered_violin_max_num_bins + 1).round(0).astype("int"))  # the indices of the feature data corresponding to each bin
    shap_min, shap_max = -10,10
    x_points = np.linspace(shap_min, shap_max, num_x_points)
    # loop through each feature and plot:
    for pos, ind in enumerate(feature_order):
        # decide how to handle: if #unique < layered_violin_max_num_bins then split by unique value, otherwise use bins/percentiles.
        # to keep simpler code, in the case of uniques, we just adjust the bins to align with the unique counts.
        feature = features[:, ind]
        unique, counts = np.unique(feature, return_counts=True)
        if unique.shape[0] <= layered_violin_max_num_bins:
            order = np.argsort(unique)
            thesebins = np.cumsum(counts[order])
            thesebins = np.insert(thesebins, 0, 0)
        else:
            thesebins = bins
        nbins = thesebins.shape[0] - 1
        # order the feature data so we can apply percentiling
        order = np.argsort(feature)
        # x axis is located at y0 = pos, with pos being there for offset
        # y0 = np.ones(num_x_points) * pos
        # calculate kdes:
        ys = np.zeros((nbins, num_x_points))
        for i in range(nbins):
            # get shap values in this bin:
            shaps = shap_values[order[thesebins[i] : thesebins[i + 1]], ind]
            # if there's only one element, then we can't
            # save kde of them: note that we add a tiny bit of gaussian noise to avoid singular matrix errors
            ys[i, :] = gaussian_kde(shaps + np.random.normal(loc=0, scale=0.001, size=shaps.shape[0]))(x_points)
            # scale it up so that the 'size' of each y represents the size of the bin. For continuous data this will
            # do nothing, but when we've gone with the unique option, this will matter - e.g. if 99% are male and 1%
            # female, we want the 1% to appear a lot smaller.
            size = thesebins[i + 1] - thesebins[i]
            bin_size_if_even = features.shape[0] / nbins
            relative_bin_size = size / bin_size_if_even
            ys[i, :] *= relative_bin_size
        # now plot 'em. We don't plot the individual strips, as this can leave whitespace between them.
        # instead, we plot the full kde, then remove outer strip and plot over it, etc., to ensure no
        # whitespace
        ys = np.cumsum(ys, axis=0)
        width = 0.8
        scale = ys.max() * 2 / width  # 2 is here as we plot both sides of x axis
        for i in range(nbins - 1, -1, -1):
            y = ys[i, :] / scale
            c = plt.get_cmap(color)(i / (nbins - 1))
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
        classes = {0:r'$p$', 1:r'$K^{+}$', 2:r'$\pi^{+}$ | $\mu^{+}$', 3:r'$e^{+}$'}
        index_offset=0
    elif charge=="neg":
        nClasses = 4
        index_offset = 4
        classes = {4:r'$\bar{p}$', 5:r'$K^{-}$', 6:r'$\pi^{-}$ | $\mu^{-}$', 7:r'$e^{-}$'}
    else:
        nClasses = 3
        classes = {0:r'$\gamma$', 1:r'$K_{L}^{0}$', 2:r'$n$'}
        index_offset=0
    fig, axes = plt.subplots(nrows=nClasses, ncols=2, figsize=(10,11))
    for i in classes.keys():
        if i == max(classes.keys()):
            BDT_x_label = "BDT SHAP Values"
            MLP_x_label = "MLP SHAP Values"
        else:
            BDT_x_label = ""
            MLP_x_label = ""
        violin(bdt_shap_vals[i],axes[i-index_offset][0],features=bdt_data,x_label=BDT_x_label)
        violin(mlp_shap_vals[i],axes[i-index_offset][1],features=mlp_data,x_label=MLP_x_label, show_label_names=False)
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
        fig.text(leftBox.x1+rel_pos_x, leftBox.y1+rel_pos_y, "SHAP Values For: " + classes[index_offset+j], ha='center', va='center', fontsize=13)
    fig.show()
    fig.savefig("/home/rdube/PID_paper/pid_mlp/bdt_plots/" + charge + "_SHAP.png",dpi=300)


