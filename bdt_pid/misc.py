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

"""

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from packaging import version
from scipy.stats import gaussian_kde

from ..utils._exceptions import DimensionError
from . import colors
from ._labels import labels

# TODO: simplify this when we drop support for matplotlib 3.9
if version.parse(matplotlib.__version__) >= version.parse("3.10"):
    ORIENTATION_KWARG = dict(orientation="horizontal")
else:
    ORIENTATION_KWARG = dict(vert=False)  # type: ignore[dict-item]


# TODO: remove unused title argument / use title argument
# TODO: Add support for hclustering based explanations where we sort the leaf order by magnitude and then show the dendrogram to the left
def violin(
    shap_values,
    features=None,
    feature_names=None,
    max_display=20,
    color="coolwarm",
    axis_color="#333333",
    title=None,
    alpha=1,
    show=True,
    sort=True,
    color_bar=True,
    plot_size="auto",
    layered_violin_max_num_bins=20,
    class_names=None,
    class_inds=None,
    color_bar_label=labels["FEATURE_VALUE"],
    cmap=colors.red_blue,
    use_log_scale=False,
):
    # convert from a DataFrame or other types
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns
        features = features.to_numpy()

    num_features = shap_values.shape[1]

    if features is not None:
        shape_msg = "The shape of the shap_values matrix does not match the shape of the provided data matrix."
        if num_features - 1 == features.shape[1]:
            shape_msg += (
                " Perhaps the extra column in the shap_values matrix is the "
                "constant offset? If so, just pass shap_values[:,:-1]."
            )
            raise DimensionError(shape_msg)
        if num_features != features.shape[1]:
            raise DimensionError(shape_msg)

    if feature_names is None:
        feature_names = np.array([labels["FEATURE"] % str(i) for i in range(num_features)])

    if use_log_scale:
        plt.xscale("symlog")

    if max_display is None:
        max_display = 20

    if sort:
        # order features by the sum of their effect magnitudes
        feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        feature_order = feature_order[-min(max_display, len(feature_order)) :]
    else:
        feature_order = np.flip(np.arange(min(max_display, num_features)), 0)

    row_height = 0.4
    if plot_size == "auto":
        plt.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
    elif type(plot_size) in (list, tuple):
        plt.gcf().set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        plt.gcf().set_size_inches(8, len(feature_order) * plot_size + 1.5)
    plt.axvline(x=0, color="#999999", zorder=-1)

    if plot_type == "violin":
        for pos in range(len(feature_order)):
            plt.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

        if features is not None:
            global_low = np.nanpercentile(shap_values[:, : len(feature_names)].flatten(), 1)
            global_high = np.nanpercentile(shap_values[:, : len(feature_names)].flatten(), 99)
            for pos, i in enumerate(feature_order):
                shaps = shap_values[:, i]
                shap_min, shap_max = np.min(shaps), np.max(shaps)
                rng = shap_max - shap_min
                xs = np.linspace(np.min(shaps) - rng * 0.2, np.max(shaps) + rng * 0.2, 100)
                if np.std(shaps) < (global_high - global_low) / 100:
                    ds = gaussian_kde(shaps + np.random.randn(len(shaps)) * (global_high - global_low) / 100)(xs)
                else:
                    ds = gaussian_kde(shaps)(xs)
                ds /= np.max(ds) * 3

                values = features[:, i]
                # window_size = max(10, len(values) // 20)
                smooth_values = np.zeros(len(xs) - 1)
                sort_inds = np.argsort(shaps)
                trailing_pos = 0
                leading_pos = 0
                running_sum = 0
                back_fill = 0
                for j in range(len(xs) - 1):
                    while leading_pos < len(shaps) and xs[j] >= shaps[sort_inds[leading_pos]]:
                        running_sum += values[sort_inds[leading_pos]]
                        leading_pos += 1
                        if leading_pos - trailing_pos > 20:
                            running_sum -= values[sort_inds[trailing_pos]]
                            trailing_pos += 1
                    if leading_pos - trailing_pos > 0:
                        smooth_values[j] = running_sum / (leading_pos - trailing_pos)
                        for k in range(back_fill):
                            smooth_values[j - k - 1] = smooth_values[j]
                    else:
                        back_fill += 1

                # Get nan values:
                nan_mask = np.isnan(values)

                # Trim the value and color range to percentiles
                vmin, vmax, cvals = _trim_crange(values, nan_mask)

                # plot the nan values in the interaction feature as grey
                plt.scatter(
                    shaps[nan_mask],
                    np.ones(shap_values[nan_mask].shape[0]) * pos,
                    color="#777777",
                    s=9,
                    alpha=alpha,
                    linewidth=0,
                    zorder=1,
                )
                # plot the non-nan values colored by the trimmed feature value
                plt.scatter(
                    shaps[np.invert(nan_mask)],
                    np.ones(shap_values[np.invert(nan_mask)].shape[0]) * pos,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    s=9,
                    c=cvals,
                    alpha=alpha,
                    linewidth=0,
                    zorder=1,
                )
                # smooth_values -= nxp.nanpercentile(smooth_values, 5)
                # smooth_values /= np.nanpercentile(smooth_values, 95)
                smooth_values -= vmin
                if vmax - vmin > 0:
                    smooth_values /= vmax - vmin
                for i in range(len(xs) - 1):
                    if ds[i] > 0.05 or ds[i + 1] > 0.05:
                        plt.fill_between(
                            [xs[i], xs[i + 1]],
                            [pos + ds[i], pos + ds[i + 1]],
                            [pos - ds[i], pos - ds[i + 1]],
                            color=colors.red_blue_no_bounds(smooth_values[i]),
                            zorder=2,
                        )

        else:
            parts = plt.violinplot(
                shap_values[:, feature_order],
                range(len(feature_order)),
                points=200,
                **ORIENTATION_KWARG,  # type: ignore[arg-type]
                widths=0.7,
                showmeans=False,
                showextrema=False,
                showmedians=False,
            )

            for pc in parts["bodies"]:  # type: ignore
                pc.set_facecolor(color)
                pc.set_edgecolor("none")
                pc.set_alpha(alpha)

    elif plot_type == "layered_violin":  # courtesy of @kodonnell
        num_x_points = 200
        bins = (
            np.linspace(0, features.shape[0], layered_violin_max_num_bins + 1).round(0).astype("int")
        )  # the indices of the feature data corresponding to each bin
        shap_min, shap_max = np.min(shap_values), np.max(shap_values)
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
                if shaps.shape[0] == 1:
                    warnings.warn(
                        f"Not enough data in bin #{i} for feature {feature_names[ind]}, so it'll be ignored."
                        " Try increasing the number of records to plot."
                    )
                    # to ignore it, just set it to the previous y-values (so the area between them will be zero). Not ys is already 0, so there's
                    # nothing to do if i == 0
                    if i > 0:
                        ys[i, :] = ys[i - 1, :]
                    continue
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
                c = (
                    plt.get_cmap(color)(i / (nbins - 1)) if color in plt.colormaps else color
                )  # if color is a cmap, use it, otherwise use a color
                plt.fill_between(x_points, pos - y, pos + y, facecolor=c, edgecolor="face")
        plt.xlim(shap_min, shap_max)

    # draw the color bar
    if (
        color_bar
        and features is not None
        and plot_type != "bar"
        and (plot_type != "layered_violin" or color in plt.colormaps)
    ):
        import matplotlib.cm as cm

        m = cm.ScalarMappable(cmap=cmap if plot_type != "layered_violin" else plt.get_cmap(color))
        m.set_array([0, 1])
        cb = plt.colorbar(m, ax=plt.gca(), ticks=[0, 1], aspect=80)
        cb.set_ticklabels([labels["FEATURE_VALUE_LOW"], labels["FEATURE_VALUE_HIGH"]])
        cb.set_label(color_bar_label, size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)  # type: ignore
        # bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
        # cb.ax.set_aspect((bbox.height - 0.9) * 20)
        # cb.draw_all()

    plt.gca().xaxis.set_ticks_position("bottom")
    plt.gca().yaxis.set_ticks_position("none")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().tick_params(color=axis_color, labelcolor=axis_color)
    plt.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=13)
    plt.gca().tick_params("y", length=20, width=0.5, which="major")
    plt.gca().tick_params("x", labelsize=11)
    plt.ylim(-1, len(feature_order))
    plt.xlabel(labels["VALUE"], fontsize=13)

    if show:
        plt.show()
"""