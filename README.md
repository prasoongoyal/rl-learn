# rl-lang-action-reward-network

This is the code for our IJCAI 2019 paper [Using Natural Language for Reward Shaping in Reinforcement Learning](https://arxiv.org/abs/1903.02020).

## Running the code:

1. Clone this repository and install dependencies using the included `requirements.txt` file. The code requires Python 3.
2. Download preprocessed data:
```
mkdir data
wget http://www.cs.utexas.edu/~pgoyal/ijcai19/train_lang_data.pkl -O ./data/train_lang_data.pkl
wget http://www.cs.utexas.edu/~pgoyal/ijcai19/test_lang_data.pkl -O ./data/test_lang_data.pkl
```
3. Run the LEARN module training and RL training using the following commands:
```
mkdir learn_model
python learn/train.py --lang_enc=onehot --save_path=./learn_model
python rl/main.py --expt_id=<expt_id> --descr_id=<descr_id> --lang_coeff=1.0 --lang_enc=onehot --model_dir=./learn_model
```

## Data

Raw data can be downloaded from http://www.cs.utexas.edu/~pgoyal/atari-lang.zip. The directories contain frames from Montezuma's revenge (downloaded from [Atari Grand Challenge dataset](http://atarigrandchallenge.com/data)). The file annotations.txt contains pairs of clip ids and natural language descriptions. The clip id is formatted as <directory_name>/<start_frame>-<end_frame>.mp4

Preprocessed data can be generated from the raw data as follows:
1. Download the [InferSent](https://github.com/facebookresearch/InferSent) model using the following command:
```
wget http://www.cs.utexas.edu/~pgoyal/ijcai19/infersent1.pkl -O ./lang_enc_pretrained/InferSent/encoder/infersent1.pkl
```
2. Download pretrained GloVe vectors (glove.6B.zip) from https://nlp.stanford.edu/projects/glove/. Put the unzipped files in `lang_enc_pretrained/glove`.

3. Run the preprocessing code as follows:
```
python scripts/preprocess_data.py
```
This will create files `train_lang_data.pkl` and `test_lang_data.pkl` in the `./data` directory.

## Acknowledgements:

The RL code is adapted from the following implementation -- https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail.
