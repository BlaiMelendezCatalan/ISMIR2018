import numpy as np
from sklearn.metrics import auc as AUC


def find_best_t(data, base='segment_based', label='overall',
                stat="f_measure"):
    """
    For a given author and dataset, get the threshold that maximizes a
    particular statistic.

    Args:
        data (dict): the statistics for each threshold.
        base (str): the type of evaluation. Either segment-based or
                    event-based.
        label (str): 'music', 'no-music' or overall.
        stat (str): stat to maximize.

    Returns:
        The chosen threshold value.
    """

    thresholds = sorted(data.keys())
    values = []
    for th in thresholds:
        values.append(data[th][0][base][label][stat])

    return thresholds[np.nanargmax(values)]


def produce_results(stats):
    """
    Select and organize the computed statistics to easily produce the plots
    for the paper.

    Args:
        stats (dict): this is a dict with the following format:
            format: stats = {author1:
                                {threshold1: [*],
                                 threshold2: [*],
                                 ...}
                             author2: {...},
                             ...
                            }
                    [*] = output of statistics.compute_statistics for one
                          author and threshold.

    Returns:
        results (dict): statistics organized to easily produce the plots
                        for the paper.
    """

    results = {}
    for a in stats.keys():
        results[a] = {}
        results[a]['segment_based'] = {}
        results[a]['confusion_matrix'] = {}
        results[a]['segment_based']['pr_curves'] = {'precisions': [],
                                                       'recalls': [],
                                                       'f_measures': [],
                                                       'thresholds': []}
        results[a]['thresholds'] = {}
        # PR-curves
        for t in sorted(stats[a].keys()):
            ds_stats = stats[a][t][0]
            precision = ds_stats['segment_based']['music']['precision']
            recall = ds_stats['segment_based']['music']['recall']
            f_measure = ds_stats['segment_based']['music']['f_measure']
            results[a]['segment_based']['pr_curves']['thresholds'].append(t)
            results[a]['segment_based']['pr_curves']['precisions'].append(precision)
            results[a]['segment_based']['pr_curves']['recalls'].append(recall)
            results[a]['segment_based']['pr_curves']['f_measures'].append(f_measure)

        # Choosing the best thresholds for the following results
        best_t_acc = find_best_t(stats[a], stat='accuracy')
        results[a]['thresholds']['best_t_acc'] = best_t_acc
        best_t_fmeasure = find_best_t(stats[a], stat='f_measure', label='music')
        results[a]['thresholds']['best_t_fmeasure'] = best_t_fmeasure

        # Confusion matrix (to compute correlation between errors and
        # music loudness)
        results[a]['confusion_matrix'] = stats[a][best_t_fmeasure][2]
        # Best f_measure and accuracy
        results[a]['segment_based']['pr_curves'][
            'f_measure'] = stats[a][best_t_fmeasure][0][
                'segment_based']['music']['f_measure']
        results[a]['segment_based']['pr_curves'][
            'accuracy'] = stats[a][best_t_acc][0][
                'segment_based']['overall']['accuracy']

    return results
