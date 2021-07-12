"""
Computes the F1-scores without clustering (Table 2 in the paper).
"""
import os
import csv
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from scipy.spatial.distance import cosine


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

    def get_similarity(self, sentence_a, sentence_b):
        vec_a = self.tf_idf_model.transform([sentence_a])
        vec_b = self.tf_idf_model.transform([sentence_b])
        vec_a = np.array(vec_a.todense()).reshape(-1) + 1e-9
        vec_b = np.array(vec_b.todense()).reshape(-1) + 1e-9
        return 1 - cosine(vec_a, vec_b)


class SupervisedSimilarityScorer:
    def __init__(self, predictions_file):
        self.score_lookup = defaultdict(dict)
        for line in open(predictions_file):
            splits = line.strip().split('\t')
            score = float(splits[-1])
            sentence_a = splits[0].strip()
            sentence_b = splits[1].strip()
            self.score_lookup[sentence_a][sentence_b] = score
            self.score_lookup[sentence_b][sentence_a] = score

    def get_similarity(self, sentence_a, sentence_b):
        return self.score_lookup[sentence_a][sentence_b]


class T2FDistanceScorer:
    def __init__(self, t2f_model):
        self.t2f_model = t2f_model
        self.doc2idx = {doc: idx for idx, doc in enumerate(t2f_model.documents)}

    def get_similarity(self, sentence_a, sentence_b):
        vec_a = self.t2f_model.document_vectors[self.doc2idx[sentence_a]]
        vec_b = self.t2f_model.document_vectors[self.doc2idx[sentence_b]]
        return 1 - cosine(vec_a, vec_b)


def eval_split(similarity_score_function, labels_file, threshold):
    all_f1_means = []
    all_f1_sim = []
    all_f1_dissim = []

    test_data = defaultdict(list)
    with open(labels_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar=None)
        for splits in csvreader:
            splits = map(str.strip, splits)
            label_topic, sentence_a, sentence_b, label = splits

            label_bin = '1' if label in ['SS', 'HS'] else '0'

            test_data[label_topic].append({'topic': label_topic,
                                           'sentence_a': sentence_a,
                                           'sentence_b': sentence_b,
                                           'label': label,
                                           'label_bin': label_bin})

    for topic in test_data:
        topic_test_data = test_data[topic]
        y_true = np.zeros(len(topic_test_data))
        y_pred = np.zeros(len(topic_test_data))

        for idx, test_annotation in enumerate(topic_test_data):
            sentence_a = test_annotation['sentence_a']
            sentence_b = test_annotation['sentence_b']
            label = test_annotation['label_bin']

            if label == '1':
                y_true[idx] = 1

            if similarity_score_function(sentence_a, sentence_b) > threshold:
                y_pred[idx] = 1

        f_sim = f1_score(y_true, y_pred, pos_label=1)
        f_dissim = f1_score(y_true, y_pred, pos_label=0)
        f_mean = np.mean([f_sim, f_dissim])
        all_f1_sim.append(f_sim)
        all_f1_dissim.append(f_dissim)
        all_f1_means.append(f_mean)

    return np.mean(all_f1_sim), np.mean(all_f1_dissim), np.mean(all_f1_means)


def final_eval(method, project_path, t2f_model=None):

    test_path_tplt = os.path.join(project_path, "datasets", "ukp_aspect", "splits",
                                  "{split}", "{mode}.tsv")
    bert_experiment_tplt = os.path.join(project_path, "bert_output", "ukp",
                                        "seed-1", "splits", "{split}",
                                        "{mode}_predictions_epoch_3.tsv")

    all_f1_sim = []
    all_f1_dissim = []
    all_f1 = []
    for split in [0, 1, 2, 3]:
        train_file = test_path_tplt.format(split=split, mode="train")
        dev_file = test_path_tplt.format(split=split, mode="dev")
        test_file = test_path_tplt.format(split=split, mode="test")

        if method == "supervised":
            dev_sim_scorer = SupervisedSimilarityScorer(bert_experiment_tplt.format(split=split,
                                                                                    mode="dev"))
            test_sim_scorer = SupervisedSimilarityScorer(bert_experiment_tplt.format(split=split,
                                                                                     mode="test"))
        elif method == "t2f":
            dev_sim_scorer = UnsupervisedSimilarityScorer(t2f_model)
            test_sim_scorer = dev_sim_scorer
        elif method == "tf_idf":
            dev_sim_scorer = TFIDFDistanceScorer(train_file)
            test_sim_scorer = dev_sim_scorer
        elif method in ["is_fasttext", "is_glove", "glove_avg", "elmo_avg", "bert_avg"]:
            dev_sim_scorer = UnsupervisedDistanceScorer(method, dev_file)
            test_sim_scorer = UnsupervisedDistanceScorer(method, test_file)
        else:
            raise ValueError("Invalid method provided.")

        best_f1 = 0
        best_threshold = 0

        for threshold_int in range(0, 20):
            threshold = threshold_int / 20
            __, __, f1_mean = eval_split(dev_sim_scorer.get_similarity,
                                         dev_file, threshold)

            if f1_mean > best_f1:
                best_f1 = f1_mean
                best_threshold = threshold

        # Evaluate on test
        f1_sim, f1_dissim, f1_mean = eval_split(test_sim_scorer.get_similarity,
                                                test_file, best_threshold)

        all_f1_sim.append(f1_sim)
        all_f1_dissim.append(f1_dissim)
        all_f1.append(f1_mean)

    print("F-Mean: %.4f" % (np.mean(all_f1)))
    print("F-sim: %.4f" % (np.mean(all_f1_sim)))
    print("F-dissim: %.4f" % (np.mean(all_f1_dissim)))
