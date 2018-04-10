# Marolt's algorithm

## Notes:

1. We cannot publish the code for Matija Maroly's algorithm as it is not public.

2. If you were to get access to the code, the modifications that we have applied are specified in Modifications section.

3. Below, we list the commands used for the different steps of the process

4. The code is in Matlab


## Modifications:

1. The input file name <inFile> and the threshold used <t> are passed to the classification function:
```
- (ln 1) function makeMusicSpeechSegmentationMirex15(inFile, outFile, pars)
+ (ln 1) function makeMusicSpeechSegmentationMirex15(inFile, outFile, t, pars)

- (ln 3) if nargin<3
+ (ln 3) if nargin<4

- (ln 17) tm=load('trainedModels');
+ (ln 17) tm=load('model/model.mat');
```

2. The model return probabilities instead of classes. The probabilities are thresholded to obtain classes:
```
- (ln 82) [~,t]=max(t);
- (ln 83) labels(st:en)=t/2
+ (ln 82) if t(1) >= pr_threshold
+ (ln 83)     labels(st:en) = 0.5;
+ (ln 84) else
+ (ln 85)     labels(st:en) = 1;
+ (ln 86) end
```

## Feature extraction:
```
extractFeaturesMirex15('../../audio/training_split', 'files/files.txt')

makeMusicSpeechSegmentationMirex15('../../audio/testing_split/<input_file_name>', '../../estimations/marolt/raw_estimations/<input_file_name>', t)
```

## Model training:
```
trainMusicSpeechMirex15('features/', 'annotations/annotations.txt', 'model.mat', <ncups>)

Move features/model.mat to model/model.mat
```

## Model testing:

Test the model using *input\_file\_name* and threshold *t*

$ makeMusicSpeechSegmentationMirex15('../../audio/testing\_split/<input_file_name>', '../../estimations/marolt/raw\_estimations/input\_file\_name', t)
