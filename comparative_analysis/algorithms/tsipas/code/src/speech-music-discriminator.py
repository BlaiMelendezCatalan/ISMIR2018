import pickle
import feat
import os
from sac.methods.sm_analysis import kernel
from sac.util import Util
import subprocess
import argparse
import glob
import numpy as np

FEATURE_PLAN = "/opt/speech-music-discrimination/featureplan"


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-file', dest='input_file', required=True)
    parser.add_argument('-t', dest='threshold', required=True)
    parser.add_argument('-o', dest='output_dir', required=True)
    parser.add_argument('-m', dest='model', required=True)
    parser.add_argument('-s', dest='scaler', required=True)
    parser.add_argument('-f', dest='temp_file', required=True)
    args = parser.parse_args()

    input_dir = os.path.split(args.input_file)[0]
    threshold = args.threshold
    output_filename = args.input_file.split('/')[-1].replace('.wav', '.txt')
    temp_file = "/tmp/" + args.temp_file

    cmd = ["ffmpeg", "-i", args.input_file, "-ar", "22050", "-ac", "1", "-acodec", "pcm_s16le",
           temp_file, "-y"]
    subprocess.check_call(cmd)

    cmd = ["yaafe", "-c", FEATURE_PLAN, "-r", "22050", temp_file]

    subprocess.check_output(cmd)

    features1 = ["zcr", "flux", "spectral_rollof", "energy_stats"]
    features2 = ["mfcc_stats"]
    features3 = ["spectral_flatness_per_band"]
    features4 = features1 + features2 + features3

    FEATURE_GROUPS = [features1, features2, features3, features4]

    peaks, convolution_values, timestamps = feat.get_combined_peaks(temp_file, FEATURE_GROUPS,
                                                                    args.scaler, kernel_type="gaussian")
    detected_segments = kernel.calculate_segment_start_end_times_from_peak_positions(peaks, timestamps)

    timestamps, feature_vectors = feat.read_features(features4, temp_file, args.scaler, scale=True)

    with open("/opt/speech-music-discrimination/model/" + args.model, 'r') as f:
        trained_model = pickle.load(f)

    frame_level_predictions_probs = trained_model.predict_proba(feature_vectors)

    frame_level_predictions = []
    for p in frame_level_predictions_probs:
        if p[0] >= float(threshold):
            frame_level_predictions.append('m')
        else:
            frame_level_predictions.append('s')

    annotated_segments = Util.get_annotated_labels_from_predictions_and_sm_segments(np.array(frame_level_predictions),
                                                                            detected_segments,
                                                                            timestamps)

    annotated_segments = Util.combine_adjacent_labels_of_the_same_class(annotated_segments)
    annotated_segments = feat.filter_noisy_labels(annotated_segments)
    annotated_segments = Util.combine_adjacent_labels_of_the_same_class(annotated_segments)

    if not os.path.exists(args.output_dir + 'threshold_' + threshold):
        os.mkdir(args.output_dir + 'threshold_' + threshold)
    Util.write_audacity_labels(annotated_segments, args.output_dir + 'threshold_' + threshold + '/' + output_filename)

    os.remove(temp_file)


if __name__ == '__main__':
    main()
