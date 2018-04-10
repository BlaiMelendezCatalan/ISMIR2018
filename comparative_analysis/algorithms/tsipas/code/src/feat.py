import numpy as np
import peakutils
from sac.methods import self_similarity
from sac.methods.sm_analysis import kernel
from sac.util import Util
from scipy import ndimage as ndi
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

GAUSSIAN_FILTER_SIGMA = 1.5
CHECKERBOARD_KERNEL_WIDTH = 10


def combine_peaks(a_peaks, b_peaks):
    for b_peak in b_peaks:
        # check if a peak already exists in the same position.
        # if the new peak has higher value -> update existing value
        if b_peak in a_peaks:
            pass
        else:
            a_peaks.append(b_peak)

    return np.sort(a_peaks).tolist()


def read_features(features, wavfile, scaler, scale=False):
    timestamps, feature_vectors = Util.read_merged_features(wavfile, features)
    if scale:
        with open("/opt/speech-music-discrimination/model/" + scaler, 'r') as f:
            scaler = pickle.load(f)
        feature_vectors = scaler.transform(feature_vectors)
    return timestamps, feature_vectors


def filter_noisy_labels(labels):
    for i in range(1, len(labels) - 1):
        lbl = labels[i]
        if lbl.end_seconds - lbl.start_seconds < 2:
            lbl.label = labels[i - 1].label
    return labels


def get_features(features, datasets_dir, n, pca=False):
    X_array = []
    y_array = []
    labels_array = []
    timestamps_array = []
    for i in xrange(n):
	print "   - Training part %d/%d" % ((i + 1), n)
        audio_file = datasets_dir + "/audio_" + str(i) + "/audio_" + str(i) + "_combined.wav"
        timestamps, feature_vectors = Util.read_merged_features(audio_file, features)
        labels = Util.read_audacity_labels(datasets_dir + "/audio_" + str(i) + "/audio_" + str(i) + "_combined.txt")
        X, y, lbls = Util.get_annotated_data_x_y(timestamps, feature_vectors, labels)
        X_array.append(X)
        y_array.append(y)
        labels_array.append(lbls)
        timestamps_array.append(timestamps)

    print "   - Computing scaler..."

    scaler = StandardScaler()
    scaler.fit(np.concatenate((X_array)))
    with open("model/scaler_.pickle", 'w') as f:
        pickle.dump(scaler, f)
    for i in xrange(len(X_array)):
        X_array[i] = scaler.transform(X_array[i])

    data = {}
    counter = 0
    for X, y, l, t in zip(X_array, y_array, labels_array, timestamps_array):
        data["x_" + str(counter)] = X
        data["y_" + str(counter)] = y
        data["labels_" + str(counter)] = l
        data["timestamps_" + str(counter)] = t
        counter += 1

    print "DONE"

    return data


def get_combined_peaks(wav_file, feature_groups, scaler, kernel_type):
    timestamps, feature_vectors = read_features(feature_groups[0], wav_file, scaler)
    sm1 = self_similarity.calculate_similarity_matrix(feature_vectors, subarray_size=6 * CHECKERBOARD_KERNEL_WIDTH)
    sm1 = ndi.filters.gaussian_filter(sm1, GAUSSIAN_FILTER_SIGMA)
    peaks1, convolution_values1 = kernel.checkerboard_matrix_filtering(sm1, CHECKERBOARD_KERNEL_WIDTH,
                                                                       kernel_type=kernel_type, thresh=0.2)

    timestamps, feature_vectors = read_features(feature_groups[1], wav_file, scaler)
    sm2 = self_similarity.calculate_similarity_matrix(feature_vectors, subarray_size=6 * CHECKERBOARD_KERNEL_WIDTH)
    sm2 = ndi.filters.gaussian_filter(sm2, GAUSSIAN_FILTER_SIGMA)
    peaks2, convolution_values2 = kernel.checkerboard_matrix_filtering(sm2, CHECKERBOARD_KERNEL_WIDTH,
                                                                       kernel_type=kernel_type, thresh=0.2)

    timestamps, feature_vectors = read_features(feature_groups[2], wav_file, scaler)
    sm3 = self_similarity.calculate_similarity_matrix(feature_vectors, subarray_size=6 * CHECKERBOARD_KERNEL_WIDTH)
    sm3 = ndi.filters.gaussian_filter(sm3, GAUSSIAN_FILTER_SIGMA)
    peaks3, convolution_values3 = kernel.checkerboard_matrix_filtering(sm3, CHECKERBOARD_KERNEL_WIDTH,
                                                                       kernel_type=kernel_type, thresh=0.2)

    timestamps, feature_vectors = read_features(feature_groups[3], wav_file, scaler)
    sm4 = self_similarity.calculate_similarity_matrix(feature_vectors, subarray_size=6 * CHECKERBOARD_KERNEL_WIDTH)
    sm4 = ndi.filters.gaussian_filter(sm4, GAUSSIAN_FILTER_SIGMA)
    peaks4, convolution_values4 = kernel.checkerboard_matrix_filtering(sm4, CHECKERBOARD_KERNEL_WIDTH,
                                                                       kernel_type=kernel_type, thresh=0.2)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(221)
    # ax1.imshow(sm1, cmap=plt.cm.gray)
    # ax1.scatter(peaks1, peaks1, marker='o', c='yellow')
    #
    # ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
    # ax2.imshow(sm2, cmap=plt.cm.gray)
    # ax2.scatter(peaks2, peaks2, marker='o', c='yellow')
    #
    # ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
    # ax3.imshow(sm3, cmap=plt.cm.gray)
    # ax3.scatter(peaks3, peaks3, marker='o', c='yellow')
    #
    # ax4 = fig.add_subplot(224, sharex=ax1, sharey=ax1)
    # ax4.imshow(sm4, cmap=plt.cm.gray)
    # ax4.scatter(peaks4, peaks4, marker='o', c='yellow')
    # plt.show()

    peaks1 = combine_peaks(peaks1, peaks2)
    peaks1 = combine_peaks(peaks1, peaks3)
    peaks1 = combine_peaks(peaks1, peaks4)

    return peaks1, convolution_values1, timestamps


def get_not_combined_peaks(datasets, feature_group, test_dataset_name, kernel_type):
    data = get_features(feature_group, datasets, pca=False)

    sm1 = self_similarity.calculate_similarity_matrix(data["x_" + test_dataset_name], metric='cosine',
                                                      subarray_size=6 * CHECKERBOARD_KERNEL_WIDTH)
    sm1 = ndi.filters.gaussian_filter(sm1, GAUSSIAN_FILTER_SIGMA)
    peaks1, convolution_values1 = kernel.checkerboard_matrix_filtering(sm1, CHECKERBOARD_KERNEL_WIDTH,
                                                                       kernel_type=kernel_type, thresh=0.10)

    return peaks1, convolution_values1, data, sm1

