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
from sklearn.metrics import f1_score
from sklearn.cluster import AgglomerativeClustering

from scorers import (TFIDFDistanceScorer, InferSentDistanceScorer,
                     SupervisedDistanceScorer,
                     T2FDistanceScorer)


def get_clustering(scorer, topics,
                   testfile, threshold):
    """"""
    unique_sentences = defaultdict(set)
    with open(testfile, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar=None)
        for splits in csvreader:
            splits = map(str.strip, splits)
            gold_topic, sentence_a, sentence_b, __ = splits
            if topics == "gold":
                topic_a = gold_topic
                topic_b = gold_topic
            elif topics == "model":
                topic_a = scorer.get_doc_topic(sentence_a)
                topic_b = scorer.get_doc_topic(sentence_b)
            if topics == "none":
                topic_a = 0   # All docs in a single aritificial topic
                topic_b = 0

            unique_sentences[topic_a].add(sentence_a)
            unique_sentences[topic_b].add(sentence_b)

    agg_cls = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                      linkage="average", distance_threshold=threshold)

    clusters = {}
    for topic in unique_sentences:
        clusters[topic] = {}
        topic_sentences = list(unique_sentences[topic])
        if len(topic_sentences) == 1:
            clusters[topic][0] = [topic_sentences[0]]
            continue

        dist_mat = scorer.get_distance_matrix(topic_sentences)

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


def best_clustering_split(split, method, topics, t2f_model,
                          test_path_tplt, project_path):
    # Evaluation files
    train_file = test_path_tplt.format(split=split, mode="train")
    dev_file = test_path_tplt.format(split=split, mode="dev")
    test_file = test_path_tplt.format(split=split, mode="test")

    if method == "supervised":
        bert_experiment_tplt = os.path.join(project_path, "bert_output", "ukp",
                                            "seed-1", "splits", "{split}",
                                            "{mode}_predictions_epoch_3_all_sentences.tsv")
        dev_sim_scorer = SupervisedDistanceScorer(bert_experiment_tplt.format(split=split,
                                                                              mode="dev"))
        test_sim_scorer = SupervisedDistanceScorer(bert_experiment_tplt.format(split=split,
                                                                               mode="test"))
    elif method == "t2f":
        dev_sim_scorer = T2FDistanceScorer(t2f_model)
        test_sim_scorer = dev_sim_scorer
    elif method == "tf_idf":
        dev_sim_scorer = TFIDFDistanceScorer(train_file)
        test_sim_scorer = dev_sim_scorer
    elif method in ["is_fasttext", "is_glove"]:
        dev_sim_scorer = InferSentDistanceScorer(method, project_path)
        test_sim_scorer = dev_sim_scorer
    elif method in ["glove_avg", "elmo_avg", "bert_avg"]:
        dev_sim_scorer = UnsupervisedDistanceScorer(method, dev_file)
        test_sim_scorer = UnsupervisedDistanceScorer(method, test_file)
    else:
        raise ValueError("Invalid method provided.")

    best_f1 = 0
    best_threshold = 0

    for threshold_int in range(0, 21):
        threshold = threshold_int / 20
        clusters = get_clustering(dev_sim_scorer,
                                  topics,
                                  dev_file, threshold)
        __, __, f1_mean = eval_split(clusters, dev_file)

        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_threshold = threshold

    # print("Best threshold on dev:", best_threshold)

    # Compute clusters on test
    clusters = get_clustering(test_sim_scorer,
                              topics,
                              test_file, best_threshold)
    return clusters


def final_eval(method, topics, project_path, t2f_model=None):
    test_path_tplt = os.path.join(project_path, "datasets", "ukp_aspect", "splits",
                                  "{split}", "{mode}.tsv")

    all_f1_sim = []
    all_f1_dissim = []
    all_f1 = []
    for split in [0, 1, 2, 3]:
        # print("\n==================")
        # print("Split:", split)
        test_file = test_path_tplt.format(split=split, mode="test")
        clusters = best_clustering_split(split, method, topics, t2f_model,
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
