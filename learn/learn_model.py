import numpy as np
import sys
import pickle
import tensorflow as tf
from random import shuffle, seed, randint
import math
from data import Data
import os
from utils_learn import *

class LearnModel(object):
    def __init__(self, mode, args=None, model_dir=None):
        if mode == 'train':
            self.args = args
            self.data = Data(args)
            self.build_graph()
            self.set_training_params()
        elif mode == 'predict':
            args_file = os.path.join(model_dir, 'args.pkl')
            model_file = os.path.join(model_dir, 'model')
            self.args = pickle.load(open(args_file, 'rb'))
            self.build_graph()
            # self.model_filename = model_filename
            saver = tf.train.Saver()
            self.sess = tf.Session()
            saver.restore(self.sess, model_file)

    def batch_norm(self, input):
        return tf.layers.batch_normalization(input, training=self.is_train)

    def conv_relu_pool(self, input, kernel_shape, bias_shape, stride):
        weights = tf.get_variable("weights", kernel_shape, \
            initializer=tf.contrib.layers.xavier_initializer(), \
            regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
        biases = tf.get_variable("biases", bias_shape, \
            initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(input, weights, strides=[1, stride, stride, 1], padding='VALID')
        pool = tf.nn.max_pool(tf.nn.relu(conv + biases), ksize=[1,2,2,1], \
            strides=[1,2,2,1], padding='VALID')
        return pool

    def text_enc_lstm(self, input_text):
        n_layers = 2
        lstm_size = self.args.lang_enc_size
        cells = []
        for _ in range(n_layers):
            cell = tf.contrib.rnn.GRUCell(lstm_size)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)
    
        output, _ = tf.nn.dynamic_rnn(
            cell, input_text, dtype=tf.float32, sequence_length=self.length)
        output_mean = tf.divide(tf.reduce_sum(output, axis=1), tf.tile(tf.expand_dims(
            tf.cast(self.length, dtype=tf.float32), axis=1), tf.constant([1, lstm_size])))
        return output_mean

    def text_enc_linear(self, input_text):
        input_size = input_text.get_shape().as_list()[-1]
        with tf.variable_scope("text_enc"):
            dense1_w = tf.get_variable("dense1_w", [input_size, self.args.lang_enc_size], \
                initializer=tf.contrib.layers.xavier_initializer(), \
                regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
            dense1_b = tf.get_variable("dense1_b", [self.args.lang_enc_size], \
                initializer=tf.constant_initializer(0.1))
            l1 = tf.add(tf.matmul(input_text, dense1_w), dense1_b)
            return l1

    def mlp(self, input_enc, output_shape, scope, dropout=1., n_layers=3):
        with tf.variable_scope(scope):
            input_size = input_enc.get_shape().as_list()[-1]
            ws = []
            bs = []
            in_dim = [input_size] + [self.args.classifier_size] * (n_layers - 1)
            out_dim = [self.args.classifier_size] * (n_layers - 1) + [output_shape]
            for l in range(n_layers):
                w_name = "dense{}_w".format(l)
                b_name = "dense{}_b".format(l)
                w = tf.get_variable(w_name, [in_dim[l], out_dim[l]], \
                        initializer=tf.contrib.layers.xavier_initializer(), \
                        regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
                b = tf.get_variable(b_name, [out_dim[l]], \
                        initializer=tf.constant_initializer(0.1))
                ws.append(w)
                bs.append(b)

        out = input_enc
        for l in range(n_layers-1):
            out = tf.nn.relu(tf.add(tf.matmul(out, ws[l]), \
                    bs[l]))
            out = self.batch_norm(out)
            out = tf.nn.dropout(out, keep_prob=dropout)
        out = tf.add(tf.matmul(out, ws[-1]), bs[-1])

        return out

    def compute_text_embedding(self):
        if self.args.lang_enc == 'onehot':
            self.lang = tf.placeholder(tf.int32, [None, MAX_SENT_LEN])
            self.length = tf.placeholder(tf.int32, None)
            emb_size = 50
            word_embeddings = tf.get_variable("word_embeddings", [ONEHOT_VOCAB_SIZE, emb_size], \
                    initializer=tf.contrib.layers.xavier_initializer())
            lang_emb = tf.nn.embedding_lookup(word_embeddings, self.lang)
            text_encoded = self.text_enc_lstm(lang_emb)
            self.lang_vec = lang_emb
        elif self.args.lang_enc == 'glove':
            self.lang = tf.placeholder(tf.float32, [None, MAX_SENT_LEN, GLOVE_EMB_DIM])
            self.length = tf.placeholder(tf.int32, None)
            text_encoded = self.text_enc_lstm(self.lang)
            self.lang_vec = self.lang
        elif self.args.lang_enc == 'infersent':
            self.lang = tf.placeholder(tf.float32, [None, INFERSENT_EMB_DIM])
            self.length = tf.placeholder(tf.int32, [None])
            text_encoded = self.text_enc_linear(self.lang)
            self.lang_vec = self.lang
        else:
            raise NotImplementedError

        return text_encoded

    def build_graph(self):
        MAX_SENT_LEN = 20
        self.weight_decay = tf.constant(self.args.weight_decay, dtype=tf.float32)
        
        self.dropout = tf.placeholder_with_default(1.0, shape=())
        self.is_train = tf.placeholder(tf.bool)

        self.action = tf.placeholder(tf.float32, [None, N_ACTIONS])

        self.action_enc = self.mlp(self.action, self.args.action_enc_size, "action_encoder")

        self.text_enc = self.compute_text_embedding()

        self.labels = tf.placeholder(tf.int32, [None], name='labels')

        self.action_text = tf.concat([self.action_enc, self.text_enc], 1)
        self.logits = self.mlp(self.action_text, 2, "classifier", dropout=self.dropout)

        label_one_hot = tf.one_hot(self.labels, 2)
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(label_one_hot, self.logits))
        self.predictions = tf.argmax(self.logits, axis=1)

        self.grad_lang = tf.gradients(self.logits[:, 1], self.lang_vec)

    def set_training_params(self):
        self.global_step = tf.train.get_or_create_global_step()
        initial_learning_rate = self.args.lr
        decay_steps = 10000
        learning_rate_decay_factor = 0.95
        initial_attention_k = 10.
        att_k_decay_factor = 0.5

        self.lr = tf.train.exponential_decay(
            initial_learning_rate, self.global_step, decay_steps, learning_rate_decay_factor, 
            staircase=True)

        opt = tf.train.AdamOptimizer(self.lr)
        grads_and_vars = opt.compute_gradients(self.loss)
        self.train_op = opt.apply_gradients(grads_and_vars, global_step=self.global_step)
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        self.init = tf.global_variables_initializer()

    def pad_seq_feature(self, seq, length):
            seq = np.asarray(seq)
            if length < np.size(seq, 0):
                    return seq[:length]
            dim = np.size(seq, 1)
            result = np.zeros((length, dim))
            result[0:seq.shape[0], :] = seq
            return result

    def pad_seq_onehot(self, seq, length):
            seq = np.asarray(seq)
            if length < np.size(seq, 0):
                    return seq[:length]
            result = np.zeros(length)
            result[0:seq.shape[0]] = seq
            return result

    def get_batch_lang_lengths(self, lang_list):
        if self.args.lang_enc == 'onehot':
            langs = []
            lengths = []
            for i, l in enumerate(lang_list):
                lengths.append(len(l))
                langs.append(np.array(self.pad_seq_onehot(l, MAX_SENT_LEN)))
            
            langs = np.array(langs)
            lengths = np.array(lengths)
            return langs, lengths
        elif self.args.lang_enc == 'glove':
            langs = []
            lengths = []
            for i, l in enumerate(lang_list):
                lengths.append(len(l))
                langs.append(np.array(self.pad_seq_feature(l, MAX_SENT_LEN)))
            
            langs = np.array(langs)
            lengths = np.array(lengths)
            return langs, lengths
        elif self.args.lang_enc == 'infersent':
            return lang_list, []
        else:
            raise NotImplementedError

    def run_batch_train(self, data, start):
        lang = self.args.lang_enc
        curr_batch_data = data[start:start+self.args.batch_size]
        action_list, lang_list, label_list = zip(*curr_batch_data)

        lang_list = np.array(lang_list)

        lang_list, length_list = self.get_batch_lang_lengths(lang_list)

        batch_dict =    {    
                            self.action: action_list, 
                            self.lang: lang_list, 
                            self.length: length_list,
                            self.dropout: self.args.dropout,
                            self.labels: label_list,
                            self.is_train: 1,
                        }

        fetches = [self.predictions, self.loss, self.train_op, self.extra_update_ops]
        pred, loss, _, _,  = self.sess.run(fetches, batch_dict)
        return loss * len(label_list), pred, label_list

    def run_batch_test(self, data, start):
        lang = self.args.lang_enc
        curr_batch_data = data[start:start+self.args.batch_size]
        action_list, lang_list, label_list = zip(*curr_batch_data)

        lang_list = np.array(lang_list)

        lang_list, length_list = self.get_batch_lang_lengths(lang_list)

        batch_dict =    {    
                            self.action: action_list, 
                            self.lang: lang_list, 
                            self.length: length_list,
                            self.labels: label_list,
                            self.is_train: 0,
                        }

        fetches = [self.predictions, self.loss]
        pred, loss = self.sess.run(fetches, batch_dict)
        return loss * len(label_list), pred, label_list

    def run_epoch(self, data, is_train):
        start = 0
        loss = 0
        labels = []
        pred = []

        while start < len(data):
            if is_train:
                batch_loss, batch_pred, batch_labels = self.run_batch_train(data, start)
            else:
                batch_loss, batch_pred, batch_labels = self.run_batch_test(data, start)

            start += self.args.batch_size
            loss += batch_loss    
            pred += list(batch_pred)
            labels += list(batch_labels)

        correct = np.sum([1.0 if x == y else 0.0 for (x, y) in zip(pred, labels)])
        return correct / len(data), loss / len(data)

    def train_network(self):
        steps_per_epoch = int(math.ceil(len(self.data.train_data) / self.args.batch_size))
        n_epochs = 50

        pickle.dump(self.args, open(os.path.join(self.args.save_path, 'args.pkl'), 'wb'))
        saver = tf.train.Saver(max_to_keep=None)

        with tf.Session() as self.sess:
            self.sess.run(self.init)

            try:
                ckpt = tf.train.get_checkpoint_state(self.args.save_path)
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                global_step = self.sess.run(self.global_step)
                epoch_start = global_step // steps_per_epoch
            except Exception as e:
                try:
                    os.mkdir(self.args.save_path)
                except:
                    pass
                epoch_start = 0

            best_val_acc = 0.0

            for epoch in range(epoch_start, n_epochs):
                shuffle(self.data.train_data)
                acc_train, loss_train = self.run_epoch(self.data.train_data, is_train=1)
                acc_valid, loss_valid = self.run_epoch(self.data.valid_data, is_train=0)

                print('Epoch: %d \t TL: %f \t VL: %f \t TA: %f \t VA: %f' % 
                    (epoch, loss_train, loss_valid, acc_train, acc_valid))

                if acc_valid > best_val_acc and self.args.save_path:
                    saver.save(self.sess, os.path.join(self.args.save_path, 'model'))
                    best_val_acc = acc_valid

    def predict(self, action_list, lang_list):
        s = np.sum(action_list)
        action_list = np.array(action_list)
        if s > 0:
            action_list /= s
        lang_list, length_list = self.get_batch_lang_lengths(lang_list)

        input_dict =    {
                            self.action: action_list,
                            self.lang: lang_list,
                            self.length: length_list,
                            self.is_train: False,
                        }
        logits = self.sess.run(self.logits, feed_dict = input_dict)
        return logits



