import numpy as np
import argparse
import tensorflow as tf
from random import seed
from learn_model import LearnModel

parser = argparse.ArgumentParser()
parser.add_argument('--n_data', type=int, default=100000, 
    help='number of samples')
parser.add_argument('--lr', type=float, default=0.0001, 
    help='initial learning rate')
parser.add_argument('--dropout', type=float, default=0.8, 
    help='dropout keep prob')
parser.add_argument('--weight_decay', type=float, default=0.0, 
    help='weight_decay')
parser.add_argument('--lang_enc', type=str, 
    help='onehot | glove | infersent')
parser.add_argument('--action_enc_size', type=int, default=128, 
    help='encoder output size for image')
parser.add_argument('--lang_enc_size', type=int, default=128, 
    help='encoder output size for language')
parser.add_argument('--classifier_size', type=int, default=128, 
    help='classifier_size')
parser.add_argument('--batch_size', type=int, default=32, 
    help='batch_size')
parser.add_argument('--data_file', default='./data/train_lang_data.pkl', 
    help='data file to use')
parser.add_argument('--actions_file', default='./data/action_labels.txt', 
    help='action file to use')
parser.add_argument('--save_path', default=None,
    help='Model save path')
args = parser.parse_args()

def main():
    np.random.seed(17)
    seed(7)
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(11)
        learn = LearnModel('train', args)
        learn.train_network()

if __name__ == '__main__':
    main()
