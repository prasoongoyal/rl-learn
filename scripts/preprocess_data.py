import numpy as np
import sys
import torch
import pickle
import string
import os

# =============================== INFERSENT ===============================

class InferSentFeatures:
    def __init__(self, lang_enc_dir, sentences):
        sys.path.insert(0, os.path.join(lang_enc_dir, 'InferSent/'))
        from models import InferSent

        version = 1
        MODEL_PATH = os.path.join(lang_enc_dir, 'InferSent/encoder/infersent%s.pkl' % version)
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                                        'pool_type': 'max', 'dpout_model': 0.0, 'version': version}
        self.model = InferSent(params_model)
        self.model.load_state_dict(torch.load(MODEL_PATH))

        W2V_PATH = os.path.join(lang_enc_dir, 'glove/glove.6B.300d.txt')
        self.model.set_w2v_path(W2V_PATH)
        self.model.build_vocab(sentences, tokenize=True)
        
    def generate_embeddings(self, sentences):
        embeddings = self.model.encode(sentences, tokenize=True)
        return embeddings

# =============================== GLOVE ===============================

class GloveFeatures:
    def __init__(self, lang_enc_dir):
        self.dictionary = {}
        with open(os.path.join(lang_enc_dir, 'glove/glove.6B.50d.txt')) as f:
            for line in f.readlines():
                line = line.strip()
                parts = line.split()
                self.dictionary[parts[0]] = np.array(list(map(eval, parts[1:])))

    def generate_embeddings(self, sentences):
        result = []
        for sentence in sentences:
            sent_emb = []
            for w in sentence.split():
                try:
                    sent_emb.append(self.dictionary[w])
                except KeyError:
                    sent_emb.append(np.zeros(50))
            result.append(sent_emb)
        return result

# =============================== ONE-HOT ===============================

class OnehotFeatures:
    def __init__(self, sentences):
        self.create_vocab(sentences)

    def update_vocab(self, vocab, words):
        for w in words:
            try:
                n = vocab[w]
                vocab[w] = n + 1
            except KeyError:
                vocab[w] = 1
        return vocab

    def create_vocab(self, sentences):
        self.vocab = {}
        for sent in sentences:
            words = sent.split()
            self.vocab = self.update_vocab(self.vocab, words)
        rare_keys = [k for k in self.vocab if self.vocab[k] < 10]
        for k in rare_keys:
            del self.vocab[k]
        self.vocab['<unk>'] = 0
        idx = 1
        for k in self.vocab.keys():
            self.vocab[k] = idx
            idx += 1

    def generate_embeddings(self, sentences):
        result = []
        for sentence in sentences:
            sent_emb = []
            for w in sentence.split():
                try:
                    sent_emb.append(self.vocab[w])
                except KeyError:
                    sent_emb.append(self.vocab['<unk>'])
            result.append(sent_emb)
        return result

# =============================== ALL DATA ===============================

def load_actions(data_dir):
    clip_to_actions = {}
    with open(os.path.join(data_dir, 'actions.txt')) as f:
        for line in f.readlines():
            line = line.strip()
            parts = line.split()
            clip_id = parts[0]
            actions = map(eval, parts[1:])
            clip_to_actions[clip_id] = list(actions)
    return clip_to_actions

def load_annotations_file(filename):
    clip_ids = []
    sentences = []
    translator = str.maketrans('', '', string.punctuation)
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip()
            (clip_id, sentence) = line.split('\t')
            sentence = sentence.lower()
            sentence = sentence.translate(translator)
            clip_ids.append(clip_id.strip())
            sentences.append(sentence)
    return clip_ids, sentences

def main(data_dir, lang_enc_pretrained, output_dir):
    train_clip_ids, train_sentences = load_annotations_file(os.path.join(
        data_dir, 'train_annotations.txt'))
    test_clip_ids, test_sentences = load_annotations_file(os.path.join(
        data_dir, 'test_annotations.txt'))

    infersent = InferSentFeatures(lang_enc_pretrained, train_sentences)
    glove = GloveFeatures(lang_enc_pretrained)
    onehot = OnehotFeatures(train_sentences)

    infersent_embeddings_train = infersent.generate_embeddings(train_sentences)
    glove_embeddings_train = glove.generate_embeddings(train_sentences)
    onehot_embeddings_train = onehot.generate_embeddings(train_sentences)

    train_data = []
    for idx in range(len(train_clip_ids)):
        clip_id = train_clip_ids[idx]
        data_pt = {}
        data_pt['clip_id'] = clip_id
        data_pt['sentence'] = train_sentences[idx]
        data_pt['glove'] = glove_embeddings_train[idx]
        data_pt['infersent'] = infersent_embeddings_train[idx]
        data_pt['onehot'] = onehot_embeddings_train[idx]
        train_data.append(data_pt)
    pickle.dump(train_data, open(os.path.join(output_dir, 'train_lang_data.pkl'), 'wb'))

    infersent_embeddings_test = infersent.generate_embeddings(test_sentences)
    glove_embeddings_test = glove.generate_embeddings(test_sentences)
    onehot_embeddings_test = onehot.generate_embeddings(test_sentences)

    test_data = []
    for idx in range(len(test_clip_ids)):
        clip_id = test_clip_ids[idx]
        data_pt = {}
        data_pt['clip_id'] = clip_id
        data_pt['sentence'] = test_sentences[idx]
        data_pt['glove'] = glove_embeddings_test[idx]
        data_pt['infersent'] = infersent_embeddings_test[idx]
        data_pt['onehot'] = onehot_embeddings_test[idx]
        test_data.append(data_pt)
    pickle.dump(test_data, open(os.path.join(output_dir, 'test_lang_data.pkl'), 'wb'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data',
        help='path to data directory')
    parser.add_argument('--output_dir', type=str, default='./data',
        help='directory where processed files will be saved')
    parser.add_argument('--lang_enc_dir', type=str, default='./lang_enc_pretrained',
        help='directory for pretrained language encoder models')
    args = parser.parse_args()
    main(args.data_dir, args.lang_enc_dir, args.output_dir)
