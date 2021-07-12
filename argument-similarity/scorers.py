""""""
from collections import defaultdict
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import torch

from InferSent.models import InferSent


class TFIDFDistanceScorer:
    def __init__(self, train_file):
        self.tf_idf_model = self.train_tf_idf(train_file)

    def train_tf_idf(self, train_file):
        train_set = set()
        with open(train_file, "r") as file:
            for line in file:
                splits = line.strip().split('\t')
                sentence_a = splits[0].strip()
                sentence_b = splits[1].strip()
                train_set.add(sentence_a)
                train_set.add(sentence_b)

        tf_idf_vectorizer = TfidfVectorizer(stop_words="english")
        tf_idf_vectorizer.fit(train_set)
        return tf_idf_vectorizer

    def get_distance_matrix(self, documents):
        tf_idf_mat = self.tf_idf_model.transform(documents)
        dist_mat = 1 - cosine_similarity(tf_idf_mat)
        return dist_mat

    def get_pairwise_distance(self, sentence_a, sentence_b):
        vec_a = self.tf_idf_model.transform([sentence_a])
        vec_b = self.tf_idf_model.transform([sentence_b])
        vec_a = np.array(vec_a.todense()).reshape(-1) + 1e-9
        vec_b = np.array(vec_b.todense()).reshape(-1) + 1e-9
        return cosine(vec_a, vec_b)


class InferSentDistanceScorer:
    def __init__(self, method, project_path):
        self.model = self.initialize_model(method, project_path)

    @staticmethod
    def initialize_model(method, project_path):
        if method == "is_glove":
            version = 1
            w2v_path = project_path + '/InferSent/fastText/crawl-300d-2M.vec'
        if method == "is_fasttext":
            version = 2
            w2v_path = project_path + '/InferSent/GloVe/glove.840B.300d.txt'
        model_path = project_path + f'/InferSent/encoder/infersent{version}.pkl'
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': version}
        infersent = InferSent(params_model)
        infersent.load_state_dict(torch.load(model_path))
        infersent.set_w2v_path(w2v_path)
        return infersent

    def get_distance_matrix(self, documents):
        self.model.build_vocab(documents, tokenize=True)
        embeddings = self.model.encode(documents, tokenize=True)
        dist_mat = 1 - cosine_similarity(embeddings)
        return dist_mat


class SupervisedDistanceScorer:
    def __init__(self, predictions_file):
        self.score_lookup = defaultdict(dict)
        for line in open(predictions_file):
            splits = line.strip().split('\t')
            score = float(splits[-1])
            sentence_a = splits[0].strip()
            sentence_b = splits[1].strip()
            self.score_lookup[sentence_a][sentence_b] = score
            self.score_lookup[sentence_b][sentence_a] = score

    def get_distance_matrix(self, documents):
        dist_mat = np.zeros((len(documents), len(documents)))
        for idx_a, sentence_a in enumerate(documents):
            for idx_b, sentence_b in enumerate(documents):
                if sentence_a == sentence_b:
                    dist_mat[idx_a, idx_b] = 0
                else:
                    dist_mat[idx_a, idx_b] = 1 - self.score_lookup[sentence_a][sentence_b]
        return dist_mat

    def get_pairwise_distance(self, sentence_a, sentence_b):
        return 1 - self.score_lookup[sentence_a][sentence_b]


class T2FDistanceScorer:
    def __init__(self, t2f_model):
        self.t2f_model = t2f_model
        self.doc2idx = {doc: idx for idx, doc in enumerate(t2f_model.documents)}

    def get_distance_matrix(self, documents):
        doc_idxs = [self.doc2idx[doc] for doc in documents]
        dist_mat = 1 - cosine_similarity(self.t2f_model.document_vectors[doc_idxs])
        return dist_mat

    def get_pairwise_distance(self, sentence_a, sentence_b):
        vec_a = self.t2f_model.document_vectors[self.doc2idx[sentence_a]]
        vec_b = self.t2f_model.document_vectors[self.doc2idx[sentence_b]]
        return cosine(vec_a, vec_b)
