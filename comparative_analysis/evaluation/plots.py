import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sed_eval.io import load_event_list
import statistics as S


AUTHOR_NAME_MAPPING = {'lidy': 'Lidy',
                       'tsipas': 'Tsipas',
                       'marolt': 'Marolt'}
COLORS = ['g', 'r', 'b', 'y']
SHAPES = ['o', '^', 's', '*']
REF_LABELS = ['music_5/0', 'music_5/1', 'music_5/2', 'music_5/3', 'music_5/4',
              'music_5/5', 'music_4/5', 'music_3/5', 'music_2/5', 'music_1/5',
              'no-music_0/5']
EST_LABELS = ['music', 'no-music']


def plot_PR_curves(results, image_name=''):

    f, axes = plt.subplots()
    for i, a in enumerate(results.keys()):
        precs = results[a]['segment_based']['pr_curves']['precisions']
        recs = results[a]['segment_based']['pr_curves']['recalls']
        acc = results[a]['segment_based']['pr_curves']['accuracy']
        f_meas = results[a]['segment_based']['pr_curves']['f_measure']
        t_acc = results[a]['thresholds']['best_t_acc']
        t_fmeas = results[a]['thresholds']['best_t_fmeasure']
        axes.plot(recs, precs, COLORS[i], label=AUTHOR_NAME_MAPPING[a])
        axes.scatter(recs, precs, c=COLORS[i], marker=SHAPES[i])

    axes.set_xlabel('% music recall', fontsize=15)
    axes.set_ylabel('% music precision', fontsize=15)
    axes.grid(color='k', linestyle='dashed', alpha=0.5)
    axes.legend()
    f.tight_layout()
    plt.ylim(0.4, 1.05)
    if image_name == '':
        f.savefig('/home/bmelendez/local_shared/PR-curves.eps')
    else:
        f.savefig('/home/bmelendez/local_shared/' + image_name)


def plot_errors_by_class(results, image_name=''):

    authors = sorted(results.keys())
    mapped_authors = []
    for a in authors:
        mapped_authors.append(AUTHOR_NAME_MAPPING[a])
    music_err = []
    fgmusic_err = []
    bgmusic_err = []
    bgmusicvl_err = []
    nomusic_err = []
    for i, a in enumerate(authors):
        cm = np.zeros(shape=(len(REF_LABELS), len(EST_LABELS)))
        t = results[a]['thresholds']['best_t_acc']
        for fname in os.listdir('../estimations/' + a + '/formatted_estimations/threshold_' + str(t) + '/'):
            ref_seg = load_event_list('../annotations/loudness_annotations/' + fname)
            est_seg = load_event_list('../estimations/' + a + '/formatted_estimations/threshold_' + str(t) + '/' + fname)
            cm += S.compute_confusion_matrix(ref_seg, est_seg, REF_LABELS, EST_LABELS)
        music_err.append((cm[0, 1] / sum(cm[0,:])))
        fgmusic_err.append((np.sum(cm[1:5, 1]) / np.sum(cm[1:5,:])))
        bgmusic_err.append((np.sum(cm[5:9, 1]) / np.sum(cm[5:9,:])))
        bgmusicvl_err.append((cm[9, 1] / sum(cm[9,:])))
        nomusic_err.append((cm[10, 0] / sum(cm[10,:])))

    ind = np.arange(len(authors)) + np.arange(len(authors)) / 10.
    width = 0.2
    f, axes = plt.subplots()
    rects1 = axes.bar(ind - 0.4, music_err, width, color='r')
    rects2 = axes.bar(ind - 0.2, fgmusic_err, width, color='b')
    rects3 = axes.bar(ind, bgmusic_err, width, color='g')
    rects4 = axes.bar(ind + 0.2, bgmusicvl_err, width, color='c')
    rects5 = axes.bar(ind + 0.4, nomusic_err, width, color='y')

    # add some text for labels, title and axeses ticks
    axes.set_ylabel("% of erroneous seconds", fontsize=15)
    axes.set_xticks(ind + width / 2)
    axes.set_xticklabels(mapped_authors, fontsize=15)
    axes.grid(color='k', linestyle='dashed', alpha=0.5)

    axes.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]),
              ('Music (5/0)', 'Fg. music (5/<5)', 'Bg. music (>1/5)', 'Very low bg. music (1/5)', 'No music (0/5)'))
    vals = axes.get_yticks()
    axes.set_yticklabels(['{:3.1f}'.format(x*100) for x in vals])
    
    f.tight_layout()
    if image_name == '':
        f.savefig('/home/bmelendez/local_shared/Errors_by_loudness.eps')
    else:
        f.savefig('/home/bmelendez/local_shared/' + image_name)
