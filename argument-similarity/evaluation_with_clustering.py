"""
Evaluate the performance on the UKP ASPECT Corpus with hierachical clustering.

Greedy hierachical clustering.
Merges two clusters if the pairwise mean cluster similarity is larger than a threshold.
Merges clusters with highest similarity first
Uses dev set to determine the threshold for supervised systems
"""
import csv
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from sklearn.cluster import AgglomerativeClustering


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

    def get_distance_matrix(self, documents):
        dist_mat = np.zeros((len(documents), len(documents)))
        for idx_a, sentence_a in enumerate(documents):
            for idx_b, sentence_b in enumerate(documents):
                if sentence_a == sentence_b:
                    dist_mat[idx_a, idx_b] = 0
                else:
                    dist_mat[idx_a, idx_b] = 1 - self.score_lookup[sentence_a][sentence_b]
        return dist_mat


class UnsupervisedSimilarityScorer:
    def __init__(self, t2f_model):
        self.t2f_model = t2f_model
        self.doc2idx = {doc: idx for idx, doc in enumerate(t2f_model.documents)}

    def get_distance_matrix(self, documents):
        doc_idxs = [self.doc2idx[doc] for doc in documents]
        dist_mat = 1 - cosine_similarity(self.t2f_model.document_vectors[doc_idxs])
        return dist_mat


def get_clustering(distance_function, pre_cluster_gold_topics,
                   testfile, threshold):
    """"""
    unique_sentences = {}
    with open(testfile, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar=None)
        for splits in csvreader:
            splits = map(str.strip, splits)
            topic, sentence_a, sentence_b, __ = splits

            if not pre_cluster_gold_topics:
                topic = 0   # All docs in a single topic
            if topic not in unique_sentences:
                unique_sentences[topic] = set()

            unique_sentences[topic].add(sentence_a)
            unique_sentences[topic].add(sentence_b)

    agg_cls = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                      linkage="average", distance_threshold=threshold)

    clusters = {}
    for topic in unique_sentences:
        clusters[topic] = {}
        topic_sentences = unique_sentences[topic]
        if len(topic_sentences) == 1:
            clusters[topic][0] = [topic_sentences[0]]
            continue

        dist_mat = distance_function(topic_sentences)

        clustering = agg_cls.fit(dist_mat)
        for cluster in clustering.labels_:
            c_doc_idxs = np.where(clustering.labels_ == cluster)[0]
            clusters[topic][cluster] = [doc for idx, doc in enumerate(topic_sentences)
                                        if idx in c_doc_idxs]

    return clusters


def eval_split(clusters, labels_file):
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
            test_data[label_topic].append(
                    {'topic': label_topic, 'sentence_a': sentence_a,
                     'sentence_b': sentence_b, 'label': label,
                     'label_bin': label_bin})

    sentences_cluster_id = {}
    for topic in clusters:
        topic_cluster = clusters[topic]
        for cluster_id in topic_cluster:
            for sentence in topic_cluster[cluster_id]:
                sentences_cluster_id[sentence] = str(topic) + "_" + str(cluster_id)

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

            if sentences_cluster_id[sentence_a] == sentences_cluster_id[sentence_b]:
                y_pred[idx] = 1

        f_sim = f1_score(y_true, y_pred, pos_label=1)
        f_dissim = f1_score(y_true, y_pred, pos_label=0)
        f_mean = np.mean([f_sim, f_dissim])
        all_f1_sim.append(f_sim)
        all_f1_dissim.append(f_dissim)
        all_f1_means.append(f_mean)

    return np.mean(all_f1_sim), np.mean(all_f1_dissim), np.mean(all_f1_means)


def best_clustering_split(split, method, t2f_model, pre_cluster_gold_topics,
                          test_path_tplt, project_path):
    # Evaluation files
    dev_file = test_path_tplt.format(split=split, mode="dev")
    test_file = test_path_tplt.format(split=split, mode="test")

    if method == "supervised":
        if pre_cluster_gold_topics:
            # HCL is performed among each (gold) topic label
            bert_experiment_tplt = os.path.join(project_path, "bert_output", "ukp",
                                                "seed-1", "splits", "{split}",
                                                "{mode}_predictions_epoch_3.tsv")
        else:
            # HCL is performed on the whole data split without topic information
            bert_experiment_tplt = os.path.join(project_path, "bert_output", "ukp",
                                                "seed-1", "splits", "{split}",
                                                "{mode}_predictions_epoch_3_no_topic_info.tsv")
        dev_sim_scorer = SupervisedSimilarityScorer(bert_experiment_tplt.format(split=split,
                                                                                mode="dev"))
        test_sim_scorer = SupervisedSimilarityScorer(bert_experiment_tplt.format(split=split,
                                                                                 mode="test"))
    elif method == "unsupervised":
        dev_sim_scorer = UnsupervisedSimilarityScorer(t2f_model)
        test_sim_scorer = dev_sim_scorer

    best_f1 = 0
    best_threshold = 0

    for threshold_int in range(0, 21):
        threshold = threshold_int / 20
        clusters = get_clustering(dev_sim_scorer.get_distance_matrix,
                                  pre_cluster_gold_topics,
                                  dev_file, threshold)
        __, __, f1_mean = eval_split(clusters, dev_file)

        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_threshold = threshold

    # print("Best threshold on dev:", best_threshold)

    # Compute clusters on test
    clusters = get_clustering(test_sim_scorer.get_distance_matrix,
                              pre_cluster_gold_topics,
                              test_file, best_threshold)
    return clusters


def final_eval(method, project_path, t2f_model=None,
               pre_cluster_gold_topics=True):
    test_path_tplt = os.path.join(project_path, "datasets", "ukp_aspect", "splits",
                                  "{split}", "{mode}.tsv")

    all_f1_sim = []
    all_f1_dissim = []
    all_f1 = []
    for split in [0, 1, 2, 3]:
        # print("\n==================")
        # print("Split:", split)
        test_file = test_path_tplt.format(split=split, mode="test")
        clusters = best_clustering_split(split, method, t2f_model,
                                         pre_cluster_gold_topics,
                                         test_path_tplt, project_path)
        f1_sim, f1_dissim, f1_mean = eval_split(clusters, test_file)

        all_f1_sim.append(f1_sim)
        all_f1_dissim.append(f1_dissim)
        all_f1.append(f1_mean)

        # print("Test-Performance on this split:")
        # print("F-Mean: %.4f" % (f1_mean))
        # print("F-sim: %.4f" % (f1_sim))
        # print("F-dissim: %.4f" % (f1_dissim))

    print("F-Mean: %.4f" % (np.mean(all_f1)))
    print("F-sim: %.4f" % (np.mean(all_f1_sim)))
    print("F-dissim: %.4f" % (np.mean(all_f1_dissim)))
