import os
from intervaltree import IntervalTree
from scipy.io.wavfile import read


def get_music_segments(raw_clf_dir, style):
    """
    Retrieves all music segments from the estimations of an algorithm.

    Args:
        raw_clf_dir (str): path to the estimations folder (does not include
                           the threshold folders)
        style (str): format of the estimations. 'detection' for lidy and marolt
                     and 'segmentation' for tsipas.

    Returns:
        d (dict): dict with the music segments for each threshold and file
    """
    d = {}
    for path, dirs, fnames in os.walk(raw_clf_dir):
        if len(fnames) != 0:
            folder = path.split('/')[-1]
            d[folder] = {}
            for fname in fnames:
                d[folder][fname] = IntervalTree()
                with open(path + '/' + fname, 'r') as f:
                    reader = f.readlines()
                    for i, row in enumerate(reader):
                        row = row.split('\t')
                        row[2] = row[2].replace('\n', '')
                        if float(row[1]) == 0.0:
                            continue
                        if row[2] == 'm':
                            gt = 'music'
                            if style == 'segmentation':
                                d[folder][fname].addi(float(row[0]),
                                                      float(row[1]),
                                                      gt)
                            elif style == 'detection':
                                d[folder][fname].addi(
                                                float(row[0]),
                                                float(row[0]) + float(row[1]),
                                                gt)

    return d


def get_formatted_gt(formatted_gt_dir, audio_dir, d):
    """
    Gives the same format to all the estimations.

    Args:
        formatted_gt_dir (str): output directory path (does not include
                                threshold folders)
        audio_dir (str): path to the audio
        d (dict): dict with the music segments for each threshold and file
    """
    for folder in d.keys():
        if not os.path.exists(formatted_gt_dir + folder + '/'):
            os.mkdir(formatted_gt_dir + folder + '/')
        for fname in d[folder].keys():
            with open(formatted_gt_dir + folder + '/' + fname, 'w') as f:
                last_end = 0.0
                sr, wav = read(audio_dir + fname.replace('.txt', '.wav'))
                duration = len(wav) / float(sr)
                lines = []
                for i, it in enumerate(sorted(d[folder][fname])):
                    # If two music intervals are separated, introduce a
                    # no-music interval in between. 
                    if round(last_end, 2) != round(it[0], 2):
                        lines.append(str(round(last_end, 2)) +\
                                     '\t' + str(round(it[0], 2)) +\
                                     '\t' + 'no-music')
                    # If start is equal or higher than the total duration,
                    # discard this row. It can be equal due to rounding.
                    if round(it[0], 2) >= duration:
                        pass
                    # If the interval's end is lower than the total duration,
                    # introduce a music interval normally
                    elif round(it[1], 2) <= duration:
                        lines.append(str(round(it[0], 2)) +\
                                     '\t' + str(round(it[1], 2)) +\
                                     '\t' + 'music')
                        last_end = round(it[1], 2)
                    # If the interval's end is equal or higher than the total
                    # duration, introduce a music interval normally that ends
                    # at total duration.
                    else:
                        lines.append(str(round(it[0], 2)) +\
                                     '\t' + str(round(duration, 2)) +\
                                     '\t' + 'music')
                        last_end = duration
                    # If it is the last interval and the end does not reach
                    # total duration, introduce a no-music interval after it that
                    # reaches total duration.
                    first_condition = (i == len(d[folder][fname]) - 1)
                    second_condition = (round(it[1], 2) < round(duration, 2))
                    if first_condition and second_condition:
                        lines.append(str(round(it[1], 2)) +\
                                     '\t' + str(round(duration, 2)) +\
                                     '\t' + 'no-music')
                # If there is no music interval, create a no-music interval
                # from start to end.
                if lines == []:
                    line = str(0.0) + '\t' + str(round(duration, 2)) +\
                           '\t' + 'no-music'
                    f.write(line)
                    continue
                # Write all intervals
                for i, line in enumerate(lines):
                    if i != 0:
                        line = '\n' + line
                    f.write(line)


def check_estimations(estimations_dir, audio_dir):
    """
    Checks if the estimations are correclt formatted.

    Args:
        estimations_dir (str): path to the estimations to check
        audio_dir (str): path to the audio
    """
    for path, dirs, fnames in os.walk(estimations_dir):
        if len(fnames) != 0:
            folder = path.split('/')[-1]
            for fname in fnames:
                sr, wav = read(audio_dir + fname.replace('.txt', '.wav'))
                duration = len(wav) / float(sr)
                with open(path + '/' + fname, 'r') as f:
                    reader = f.readlines()
                    if len(reader) == 0:
                        raise ValueError("Empty file: {0}".format(fname))
                    last_end = 0.0
                    for i, row in enumerate(reader):
                        if i == 0 and float(row[0]) != 0.0:
                            raise ValueError("Start != 0.0 ({0})".format(fname)
                        row = row.split('\t')
                        row[2] = row[2].replace('\n', '')
                        if float(row[0]) != last_end:
                            raise ValueError("Gap found ({0})".format(fname))
                        last_end = float(row[1])
                    if last_end != round(duration, 2):
                        raise ValueError("End not reached ({0})".format(fname))
