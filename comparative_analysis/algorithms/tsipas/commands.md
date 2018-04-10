Modifications:
**************

1) Replaced model.predict by model (sklearn SVC) predict_proba and added thresholding:

    - frame_level_predictions = trained_model.predict(feature_vectors)
    + frame_level_predictions_probs = trained_model.predict_proba(feature_vectors)
    + frame_level_predictions = []
    + for p in frame_level_predictions_probs:
    +     if p[0] >= float(threshold):
    +         frame_level_predictions.append('m')
    +     else:
    +         frame_level_predictions.append('s')


Getting ready:
**************

$ Download the training and testing splits from TODO(link)

$ Extract files into ../../audio/training_split and ../../audio/testing_split

$ (sudo) docker pull blaimelcat/comparative_analysis_tsipas_algorithm

$ (sudo) docker run -it --name bmc_ismir2018_tsipas_alg -v </abs/path/to/audio>:/audio/ blaimelcat/comparative_analysis_tsipas_algorithm bash

$ cd /audio/training_split

$ mv `ls | head -1000` /opt/speech-music-discrimination/training_dataset/audio_0/audio/

$ mv `ls | head -1000` /opt/speech-music-discrimination/training_dataset/audio_1/audio/

$ mv `ls | head -1000` /opt/speech-music-discrimination/training_dataset/audio_2/audio/

$ mv `ls | head -1000` /opt/speech-music-discrimination/training_dataset/audio_3/audio/


Create training audio files:
****************************

$ cd /opt/speech-music-discrimination/training_dataset/audio_0/audio/; python combine.py

$ cd /opt/speech-music-discrimination/training_dataset/audio_1/audio/; python combine.py

$ cd /opt/speech-music-discrimination/training_dataset/audio_2/audio/; python combine.py

$ cd /opt/speech-music-discrimination/training_dataset/audio_3/audio/; python combine.py


Feature extraction and model training:
**************************************

# Extract features from the training audio files and store them in the same
# folder. Then, load the features and train the model. The model is stored
# in model/.

$ cd /opt/speech-music-discrimination/; python create_model.py


Model testing:
**************

# Test a model with an <input_file> using threshold <t> (float). The result is
# stored in <output_dir>. The threshold should fullfil 0 < t < 1.

$ cd /opt/speech-music-discrimination/

$ python speech-music-discriminator.py --input-file /audio/testing_split/<input_file_name> -t <t> -o ../../estimations/tsipas/raw_estimations/ -m model/model.pickle -s model/scaler.pickle -f temp.wav
