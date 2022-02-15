import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from IPython.display import Markdown, display
from aif360.metrics import ClassificationMetric

def default_preprocessing(df):
    df['credit'] = df['credit'].replace({1.0: 0, 2.0: 1})
    return df

def test(dataset, model, thresh_arr, unprivileged_groups, privileged_groups):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0
    
    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

        metric_arrs['thres'].append(thresh_arr[0])
        tpr = metric.true_positive_rate()
        tnr = metric.true_negative_rate()
        metric_arrs['bal_acc'].append(1-((tpr+tnr)/2))
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
    
    return metric_arrs


def plot(x, x_name, y_left, y_left_name, y_left_lim):
    fig, ax1 = plt.subplots(figsize=(10,7))
    ax1.plot(x, y_left)
    ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
    ax1.set_ylabel(y_left_name, color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.set_ylim(y_left_lim[0], y_left_lim[1])

    # ax2 = ax1.twinx()

    best_ind = np.argmax(y_left)
    # print(thresh_arr[best_ind])
    ax1.axvline(np.array(x)[best_ind], color='k', linestyle=':')
    ax1.grid(True)


def plot_acc_fair(x, x_name, y_left, y_left_name, y_right, y_right_name, y_left_lim, y_right_lim, **kwargs):
    new_thres = kwargs.get('thres', None)
    
    fig, ax1 = plt.subplots(figsize=(10,7))
    ax1.plot(x, y_left)
    ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
    ax1.set_ylabel(y_left_name, color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.set_ylim(y_left_lim[0], y_left_lim[1])

    ax2 = ax1.twinx()
    ax2.plot(x, y_right, color='r')
    ax2.set_ylabel(y_right_name, color='r', fontsize=16, fontweight='bold')
    ax2.set_ylim(y_right_lim[0], y_right_lim[1])

    best_ind = np.argmax(y_left)
    ax2.axvline(np.array(x)[best_ind], color='k', linestyle=':')
    if new_thres is not None:
        ax2.axvline(new_thres, color='k', linestyle=':')
    else: 
        pass
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)