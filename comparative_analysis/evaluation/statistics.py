import os
from dcase_util.containers import MetaDataContainer
from sed_eval.sound_event import SegmentBasedMetrics, EventBasedMetrics
from sed_eval.io import load_event_list
from sed_eval.util.event_list import unique_event_labels
from sed_eval import metric
from intervaltree import IntervalTree
from utils import run_mp
import numpy as np


MAPPING_DICT = {'music': 'music',
                'fg-music': 'music',
                'bg-music': 'music',
                'no-music': 'no-music'}
REF_LABELS = ['music', 'fg-music', 'bg-music', 'no-music']
EST_LABELS = ['music', 'no-music']


def map_labels(container, mapping):
    """
    Maps the event_labels in the container using the mapping dict.

    Args:
        container (MetaDataContainer): contains the events to be mapped
        mapping (dict): contains the mapping relationships.

    Returns:
        container (MetaDataContainer): mapped container.
    """

    mapped_container = MetaDataContainer()
    for c in container:
        mapped_container.append(MetaDataContainer().item_class(
                                    {'event_label': mapping[c.event_label],
                                     'offset': c.offset,
                                     'onset': c.onset}))

    return mapped_container


def collapse_labels(container):
    """
    Merges contiguous events in container that share the same event_label.

    Args:
        container (MetaDataContainer): contains the events to be collapsed.

    Returns:
        collapsed_container (MetaDataContainer): collapsed container.
    """

    collapsed_container = MetaDataContainer()
    last_class = ''
    for i, c in enumerate(container):
        if last_class != c.event_label:
            if i != 0:
                offset = c.onset
                collapsed_container.append(MetaDataContainer().item_class(
                                                   {'event_label': last_class,
                                                    'offset': offset,
                                                    'onset': onset}))
            onset = c.onset
            last_class = c.event_label
        if i == len(container) - 1:
            collapsed_container.append(collapsed_container.item_class(
                                               {'event_label': last_class,
                                                'offset': c.offset,
                                                'onset': onset}))

    return collapsed_container


def reducer(str1, str2):
    """
    Joins two strings in alphabetical order using '__'.

    Args:
        str1 (str): first string.
        str2 (str): second string.
    Returns:
        Joined strings.
    """

    strs = [str1, str2]
    strs = sorted(strs)
    return strs[1] + '__' + strs[0]


def compute_confusion_matrix(ref_seg, est_seg, ref_labels, est_labels):
    """
    Computes the confusion matrix for one file.

    Args:
        ref_seg (MetaDataContainer): contains the reference events.
        est_seg (MetaDataContainer): contains the estimated events.
        ref_labels (list): unique labels used in the reference.
        est_labels (list): unique labels used in the estimation.

    Returns:
        cm (numpy array): confusion matrix.
    """

    its = IntervalTree()
    for r in ref_seg:
        its.addi(r['onset'], r['offset'], 'ref_' + r['event_label'])
    for e in est_seg:
        its.addi(e['onset'], e['offset'], 'est_' + e['event_label'])

    its.split_overlaps()
    its.merge_equals(data_reducer=reducer)

    cm = np.zeros(shape=(len(ref_labels), len(est_labels)))
    for it in its:
        r, e = it[2].split('__')
        r = r.replace('ref_', '')
        e = e.replace('est_', '')
        cm[ref_labels.index(r), est_labels.index(e)] += it[1] - it[0]

    return cm


def compute_file_statistics(args):
    """
    Computes the statistics for one file.

    Args:
        args (list): necessary data. Supplied by run_mp function.

    Returns:
        ref_file_name (str): name of the reference file. It shares the name
                             with its corresponding wav file.
        metrics (dict): contains the SegmentBasedMetrics and EventBasedMetrics
                        objects, which include the intermediate statistics (tp,
                        tn, fp, ...), for one file.
        results (dict): contains the statistics for the segment-based and the
                        event-based evaluation as well as the confusion matrix
                        for one file.
    """

    (ref_file,
     est_file,
     ref_labels,
     est_labels,
     mapping,
     time_resolution,
     t_collar,
     percentage_of_length,
     eval_onset,
     eval_offset) = args

    ref_seg = load_event_list(ref_file)
    ref_seg_map = collapse_labels(map_labels(ref_seg, mapping))
    est_seg = load_event_list(est_file)

    seg_met = SegmentBasedMetrics(est_labels, time_resolution=time_resolution)
    ev_met = EventBasedMetrics(est_labels,
                               t_collar=t_collar,
                               percentage_of_length=percentage_of_length,
                               evaluate_onset=eval_onset,
                               evaluate_offset=eval_offset)

    seg_met.evaluate(ref_seg_map, est_seg)
    ev_met.evaluate(ref_seg_map, est_seg)
    metrics = {'event_based': ev_met, 'segment_based': seg_met}

    results = {}
    results['segment_based'] = seg_met.results()
    results['event_based'] = ev_met.results()
    results['cm'] = compute_confusion_matrix(ref_seg, est_seg,
                                             ref_labels, est_labels)

    ref_file_name = ref_file.split('/')[-1]

    return (ref_file_name, metrics, results)


def get_stats_by_file_and_dataset_int_stats(mp_results, labels):
    """
    Saves the results of each file in a stats_by_file and aggregates the
    intermediate statistics (tp, fp, fn, ...) for the whole dataset.

    Args:
        mp_results (list): contains the results of the function
                           compute_file_statistics for all files in the
                           dataset.
        labels (list): unique labels used in the estimation.

    Returns:
        stats_by_file (dict): contains the results for each file in the
                                dataset.
        ds_int_stats (dict): contains the aggregated intermediate statistics
                          for the whole dataset.
    """

    stats_by_file = {}
    ds_int_stats = {}
    ds_int_stats['segment_based'] = {}
    ds_int_stats['event_based'] = {}
    for (file_name, metrics, file_res) in mp_results:
        stats_by_file[file_name] = file_res
        for base in ds_int_stats.keys():
            for label in labels:
                if label not in ds_int_stats[base].keys():
                    ds_int_stats[base][label] = {}
                # Aggregate int_stats (tp, tn, fp, fn) for segment- and
                # event-based class-wise stats for the whole dataset.
                for stat in metrics[base].class_wise[label].keys():
                    if stat not in ds_int_stats[base][label].keys():
                        ds_int_stats[base][label][stat] = 0.
                    ds_int_stats[base][label][
                              stat] += metrics[base].class_wise[label][stat]
            # Aggregate int_stats (tp, tn, fp, fn) for segment- and
            # event-based overall stats for the whole dataset.
            if 'overall' not in ds_int_stats[base].keys():
                ds_int_stats[base]['overall'] = {}
            for stat in metrics[base].overall.keys():
                if stat not in ds_int_stats[base]['overall'].keys():
                    ds_int_stats[base]['overall'][stat] = 0.
                ds_int_stats[base]['overall'][
                          stat] += metrics[base].overall[stat]

    return stats_by_file, ds_int_stats


def get_dataset_stats(int_stats, base, label):
    """
    Computes the final segment-based and event-based statistics for the whole
    dataset using the intermediate statistics.

    Args:
        int_stats (dict): intermediate statistics of the whole dataset.
        base (str): the type of evaluation. Either segment_based or
                    event_based.
        label (str): one of the unique labels used in the estimation.

    Returns:
        stats (dict): final segment-based and event-based statistics for the
                      whole dataset.
    """

    stats = {}
    stats['precision'] = metric.precision(
                            Ntp=int_stats['Ntp'],
                            Nsys=int_stats['Nsys'])
    stats['recall'] = metric.recall(
                            Ntp=int_stats['Ntp'],
                            Nref=int_stats['Nref'])
    stats['f_measure'] = metric.f_measure(
                            precision=stats['precision'],
                            recall=stats['recall'])
    if base == 'segment_based':
        stats['sensitivity'] = metric.sensitivity(
                                Ntp=int_stats['Ntp'],
                                Nfn=int_stats['Nfn'])
        stats['specificity'] = metric.specificity(
                                Ntn=int_stats['Ntn'],
                                Nfp=int_stats['Nfp'])
        if label == 'overall':
            stats['accuracy_bmat'] = int_stats['Ntp'] / int_stats['Nref']
            stats['accuracy'] = metric.accuracy(
                                        Ntp=int_stats['Ntp'],
                                        Ntn=int_stats['Ntn'],
                                        Nfp=int_stats['Nfp'],
                                        Nfn=int_stats['Nfn'])
    elif base == 'event_based':
        stats['deletion_rate'] = metric.deletion_rate(
                                    Nref=int_stats['Nref'],
                                    Ndeletions=int_stats['Nfn'])
        stats['insertion_rate'] = metric.insertion_rate(
                                    Nref=int_stats['Nref'],
                                    Ninsertions=int_stats['Nfp'])
        stats['error_rate'] = metric.error_rate(
                                deletion_rate_value=stats['deletion_rate'],
                                insertion_rate_value=stats['insertion_rate'])
        if label == 'overall':
            stats['substitution_rate'] = metric.substitution_rate(
                                    Nref=int_stats['Nref'],
                                    Nsubstitutions=int_stats['Nsubs'])

    return stats


def compute_statistics(ref_dir, est_dir, ref_labels=REF_LABELS,
                       est_labels=EST_LABELS, mapping=MAPPING_DICT,
                       time_resolution=0.001, t_collar=0.2,
                       percentage_of_length=0.5, eval_onset=True,
                       eval_offset=True, ncpus=1):
    """
    Computes statistics for the whole dataset and for each file as well as the
    confusion matrix for the whole dataset.

    Args:
        ref_dir (str): directory of the references.
        est_dir (str): directory of the estimations.
        ref_labels (list): unique labels used in the reference.
        est_labels (list): unique labels used in the estimation.
        mapping (dict): contains the mapping relationships for the references.
        time_resolution (float): time interval used in the segment-basd
                                 evaluation. The comparison between reference
                                 and estimation is done by segments of this
                                 length.
        t_collar (float): time interval used in the event-based evaluation.
                          estimated events are correct if they fall inside
                          a range specified by t_collar from a reference event.
        percentage_of_length (float): percentage of the length within which the
                                      estimated offset has to be in order to be
                                      consider valid estimation (form sed_eval)
        eval_onset (bool): Use onsets in the event-based evaluation.
        eval_offset (bool): Use offets in the event-based evaluation.
        ncpus (int): Number of CPU to use.

    Returns:
        ds_stats (dict): statistics of the whole dataset.
        stats_by_file (dict): statistics of each file.
        cm (numpy array): confusion matrix.
    """

    ref_list = sorted(os.listdir(ref_dir))
    est_list = sorted(os.listdir(est_dir))
    args = []
    for ref, est in zip(ref_list, est_list):
        assert ref == est, ("File names do not coincide."
                            "ref: %s, est: %s") % (ref, est)
        args.append([ref_dir + ref,
                     est_dir + est,
                     ref_labels,
                     est_labels,
                     mapping,
                     time_resolution,
                     t_collar,
                     percentage_of_length,
                     eval_onset,
                     eval_offset])
    mp_results = run_mp(compute_file_statistics, args, ncpus)

    (stats_by_file,
     ds_int_stats) = get_stats_by_file_and_dataset_int_stats(mp_results,
                                                             est_labels)

    ds_stats = {}
    for base in ds_int_stats.keys():
        ds_stats[base] = {}
        for label in ds_int_stats[base].keys():
            ds_stats[base][label] = get_dataset_stats(
                                            ds_int_stats[base][label],
                                            base,
                                            label)

    cm = np.zeros(shape=(len(ref_labels), len(est_labels)))
    for fname in stats_by_file.keys():
        cm += stats_by_file[fname]['cm']

    return ds_stats, stats_by_file, cm
