# Reproducibility repository
With this repository we aim to make the ISMIR 2018 paper *Music detection in broadcast audio recordings: a non-binary approach with relative loudness annotations*  as **reproducible** as possible.

## Getting started:

1. Clone this repository
```
git clone https://github.com/BlaiMelendezCatalan/ISMIR2018.git
```

2. Download audio (and optionally Lidy's features):

Object | File name | Description | Link
------ | --------- | ----------- | ----
Audio | audio.zip | Audio used to train and test the algorithms of the comparative analysis | https://zenodo.org/record/1216054
Lidy's Features | features.npz | Features used for training in Lidy's algorithm | https://zenodo.org/record/1216062

3. Extract audio.zip, estimations.zip, annotations.zip and features.npz to the corresponding folder:

File name | Extract to
--------- | ----------
audio.zip | comparative\_analysis/
estimations.zip | comparative\_analysis/
annotations.zip | comparative\_analysis/
features.npz | comparative\_analysis/algorithms/lidy/features/

NOTE: estimations.zip and annotations.zip can be found in comparative\_analysis/

## Feature extraction and model training and testing

4. Follow the READMEs in the folder of each author (in comparative\_analysis/algorithms/) to obtain their estimations for the testing dataset. The complete process including feature extraction, the training of the model and its testing can only be done for the Tsipas algorithm. For Lidy's and Marolt's algorithms only the testing part is available as the algorithms' code is not public. The result of the process for each *author* is already available in comparative\_analysis/estimations/*author*/raw\_estimations/.

## Format estimations

5. Follow the README in comparative\_analysis/format\_estimations/ to format the estimations of each algorithm. The result of the formatting is already available in comparative\_analysis/estimations/*author*/formatted\_estimations/

## Plot results

6. Follow the README in comparative\_analysis/evaluation/ to obtain the plots in the paper.

