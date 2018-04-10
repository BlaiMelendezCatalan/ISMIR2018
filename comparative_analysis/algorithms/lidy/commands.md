Notes:
******
1) We cannot publish the code for Thomas Lidy's algorithm as it is a private repository.

2) If you were to get access to the code, we have used branch master, with mainly the modifications specified in Modifications section.

3) Below, we list the commands used for the different steps of the process


Modifications:
**************

1) The input file name (in_file) and the threshold used (threshold) are passed to the classification function:

    -def classify_segments(model, spectrogram, seg_to_analyze, frames, time_per_frame, ...):
    +def classify_segments(model, spectrogram, seg_to_analyze, frames, time_per_frame, in_file, threshold, ...):

2) The model return probabilities instead of classes:

    - predictions = model.predict_by_value(...)
    + probabilities = model.predict_by_value_get_probs(...)
    + predictions = []
    + for p in probabilities:
    +     if p[0] >= threshold:
    +         predictions.append(0)
    +     else:
    +         predictions.append(1)


Feature extraction:
*******************

# Compute the features (features.npz) of all the files in files.txt and stores
# them in ../features/. The parameters used in the feature extraction are
# stored in the same folder as features_audio_param.json.

$ python -u rp_convnet_analysis.py ../files/files.txt ../features/ --all

NOTE: ../features/thetures_audio_param.json contains the parameters used for
      the feature extraction.


Model training:
***************

# Train a CNN using features.npz and the corresponding annotations listed in
# annotations.txt. The model is stored in ../model/.

$ python -u rp_convnet.py ../features/features.npz ../annotations/annotations.txt ../model/ - -

NOTE: features.npz can be downloaded from TODO(LINK)


Model testing:
**************

# Test a model with an <input_file> using threshold <t> (float). The threshold
# should fullfil 0 < t < 1.

$ python -u rp_conv_stream_predict.py ../model/best_model.pkl - <input_file> ../../estimations/lidy/raw_estimations/ <t>
