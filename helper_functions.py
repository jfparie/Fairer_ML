import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.special import expit
from collections import defaultdict
from IPython.display import Markdown, display
from aif360.metrics import ClassificationMetric
from sklearn.preprocessing import StandardScaler

def default_preprocessing(df):
    df['credit'] = df['credit'].replace({1.0: 0, 2.0: 1})
    return df


def test_model(dataset, coef, intercept, thresh_arr, unprivileged_groups, privileged_groups):
    try:
        # sklearn classifier
        y_val_pred_prob = predict_prob(dataset, coef, intercept)
        pos_ind = 0
    except AttributeError:
        print("AttributeError")
        
    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

        metric_arrs['thres'].append(thresh)
        tpr = metric.true_positive_rate()
        tnr = metric.true_negative_rate()
        metric_arrs['bal_acc'].append(1-((tpr+tnr)/2))
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
    
    return metric_arrs


def predict_prob(dataset, coef, intercept):
    
    # scale input data
    scaler = StandardScaler()
    X = scaler.fit_transform(dataset.features)
    
    # multiply coefficients with data
    scores = safe_sparse_dot(X, coef.T, dense_output=True) + intercept
    
    # logistic transformation
    prob = expit(scores, out=scores)
    y_val_pred_prob = np.vstack([1 - prob, prob]).T
    
    return y_val_pred_prob


## LR model (from sklearn library)

def safe_sparse_dot(a, b, *, dense_output=False):
    """Dot product that handle the sparse matrix case correctly.
    Parameters
    ----------
    a : {ndarray, sparse matrix}
    b : {ndarray, sparse matrix}
    dense_output : bool, default=False
        When False, ``a`` and ``b`` both being sparse will yield sparse output.
        When True, output will always be a dense array.
    Returns
    -------
    dot_product : {ndarray, sparse matrix}
        Sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
    """
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b

    if (
        sparse.issparse(a)
        and sparse.issparse(b)
        and dense_output
        and hasattr(ret, "toarray")
    ):
        return ret.toarray()
    return ret


def plot(x, x_name, y_left, y_left_name, y_left_lim, **kwargs):
    color_dark = "#0043CE"
    color_light = "#777777"
    
    # plotting
    fig, ax1 = plt.subplots(figsize=(10,7))
    ax1.plot(x, y_left, color=color_dark)
    
    # if second plot available
    second_plot = kwargs.get('second_plot', None)
    
    if second_plot is not None:
        ax1.plot(x, second_plot, color=color_light)
    else:
        pass
    
    # axes
    ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
    ax1.set_ylabel(y_left_name, fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.set_ylim(y_left_lim[0], y_left_lim[1])
    ax1.grid(True)


    # best threshold
    best_ind = np.argmax(y_left)
    ax1.axvline(np.array(x)[best_ind], color=color_dark, linestyle=':')
    if second_plot is not None:
        best_ind2 = np.argmax(second_plot)
        ax1.axvline(np.array(x)[best_ind2], color=color_light, linestyle=':')
        
        # legend
        ax1.legend(['Strategy 1','Strategy 2'])
    else:
        pass


def plot_acc_fair(x, x_name, y_left, y_left_name, y_right, y_right_name, y_left_lim, y_right_lim, **kwargs):
    new_thres = kwargs.get('thres', None)
    color = kwargs.get('color', None)
    if color == 'color_dark':
        color = '#0043CE'
    else:
        color = '#777777'
    
    fig, ax1 = plt.subplots(figsize=(10,7))
    ax1.plot(x, y_left, color=color)
    ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
    ax1.set_ylabel(y_left_name, color=color, fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.set_ylim(y_left_lim[0], y_left_lim[1])
    ax1.yaxis.set_label_coords(-0.1, 0.5)

    ax2 = ax1.twinx()
    ax2.plot(x, y_right, color='r')
    ax2.set_ylabel(y_right_name, color='r', fontsize=16, fontweight='bold')
    ax2.set_ylim(y_right_lim[0], y_right_lim[1])
    ax2.yaxis.set_label_coords(1.1, 0.5)

    best_ind = np.argmax(y_left)
    ax2.axvline(np.array(x)[best_ind], color='k', linestyle=':')
    if new_thres is not None:
        for thres in new_thres:
            ax2.axvline(thres, color='k', linestyle=':')
    else: 
        pass
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)
    
    ax1.legend(['Strategy 2',''])