# Tsipas' algorithm

## Getting ready:
```
(sudo) docker pull blaimelcat/comparative_analysis_tsipas_algorithm

(sudo) docker run -it --name bmc_ismir2018_tsipas_alg -v </abs/path/to/audio>:/audio/ blaimelcat/comparative_analysis_tsipas_algorithm bash
```
(Inside docker container)
```
apt-get update

pip install --upgrade pip

pip install tqdm

apt-get install sox

cd /audio/training_split

mv `ls | head -1000` /opt/speech-music-discrimination/training_dataset/audio_0/audio/

mv `ls | head -1000` /opt/speech-music-discrimination/training_dataset/audio_1/audio/

mv `ls | head -1000` /opt/speech-music-discrimination/training_dataset/audio_2/audio/

mv `ls | head -1000` /opt/speech-music-discrimination/training_dataset/audio_3/audio/
```

## Create training audio files:
```
cd /opt/speech-music-discrimination/training_dataset/audio_0/; python combine.py

cd /opt/speech-music-discrimination/training_dataset/audio_1/; python combine.py

cd /opt/speech-music-discrimination/training_dataset/audio_2/; python combine.py

cd /opt/speech-music-discrimination/training_dataset/audio_3/; python combine.py
```

NOTE: This takes long and if possible should be run in parallel

## Feature extraction and model training:

Extract features from the training audio files and store them in the same folder. Then, load the features and train the model. The model is stored in model/.
```
cd /opt/speech-music-discrimination/; python create_model.py
```

## Model testing:

Test a model with an *input_file* using threshold *t* (float). The threshold should fullfil 0 < t < 1.
```
cd /opt/speech-music-discrimination/

python speech-music-discriminator.py --input-file /audio/testing_split/input_file_name -t t -o ../../estimations/tsipas/raw_estimations/ -m model/model.pickle -s model/scaler.pickle -f temp.wav
```

NOTE: Run this commands for each *input_file* and *threshold*.
