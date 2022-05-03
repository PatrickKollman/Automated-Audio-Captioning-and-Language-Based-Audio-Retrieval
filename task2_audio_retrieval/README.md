# Audio Retrieval Task


## Preparing the Data

To run this code, you will first need to clone this repo and prepare the necessary data. Follow the data preparation steps [here](https://github.com/xieh97/dcase2022-audio-retrieval). At the end of this step, you should have a "Clotho.v2.1/" directory with at least the following files (there will be more):

(1) audio_info.pkl

(2) audio_logmel.hdf5

(3) development_captions.json

(4) validation_captions.json

(5) evaluation_captions.json

(6) vocab_info.pkl

(7) word2vec_emb.pkl

Additionally, download the file "vgg.pt" from [here](https://drive.google.com/file/d/1bqWmBGBJXQHSiIdCodK48zdDXNz03TFx/view?usp=sharing) and save it under "saved_data/".

## Running an Experiment

To run one of the experiments described in the paper,  we have provided training and evaluation functions.

For training, use the following command

```
> python3 <experiment-name>.py
```

To perform evaluation (computing the R1, R5, R10, mAP10 metrics), set the "model_path" variable in line 20 of "<experiment-name>_test.py" and run the code using the following command:

```
> python3 <experiment-name>_test.py
```

There are 5 different experiments that can be run using the above method:

(1) baseline : The baseline CRNN model

(2) lstm : Replaces the GRU cell with LSTM

(3) vgg : Uses VGGish features instead of logmel

(4) shared_space : Adds a FC layer to learn joint embeddings for the text and audio

(5) 