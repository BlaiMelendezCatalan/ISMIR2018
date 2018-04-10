# Reproducibility repository
With this repository we aim to make the ISMIR 2018 paper *Music detection in broadcast audio recordings: a non-binary approach with relative loudness annotations*  as **reproducible** as possible.

## Getting started:

1. Clone this repository
```
git clone https://github.com/BlaiMelendezCatalan/ISMIR2018.git
```

2. Download audio, estimations, annotations and features and extract them into the corresponding folders:

Object | Description | Link | Extract to
------ | ----------- | ---- | ----------
Complete dataset | All audio plus annotations | TODO(link) | -
Training dataset | Audio used to train the algorithms of the comparative analysis | TODO(link) | comparative\_analysis/audio/training\_split/
Testing dataset | Audio used to test the algorithms of the comparative analysis | TODO(link) | comparative\_analysis/audio/testing\_split/
Estimations | Result of running all algorithms against the testing dataset | TODO(link) | comparative\_analysis/
Annotations | Annotations for the testing dataset with and without loudness information | TODO(link) | comparative\_analysis/
Lidy's Features | Features used for training in Lidy's algorithm | TODO(link) | comparative\_analysis/algorithms/lidy/features/

3. Follow the READMEs in the folder of each author (in comparative\_analysis/algorithms/) to obtain their estimations for the testing dataset. The complete process including feature extraction, the training of the model and its testing can only be done for the Tsipas algorithm. For Lidy's and Marolt's algorithms only the testing part is available as the algorithms' code is not public. The result of the process for each *author* is already available in comparative\_analysis/estimations/*author*/raw\_estimations/.

4. Follow the README in comparative\_analysis/format\_estimations/ to format the estimations of each algorithm. The result of the formatting is already available in comparative\_analysis/estimations/*author*/formatted\_estimations/

5. Follow the README in comparative\_analysis/evaluation/ to obtain the plots in the paper.

