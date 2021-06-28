"""
Evaluates the performance on the UKP ASPECT Corpus with hierachical clustering.

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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering


def hclustering(t2f_model, testfile, eval_method, threshold):
    unique_sentences = set()
    with open(testfile, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar=None)
        for splits in csvreader:
            splits = map(str.strip, splits)
            gold_topic, sentence_a, sentence_b, __ = splits
            unique_sentences.add((sentence_a, gold_topic))
            unique_sentences.add((sentence_b, gold_topic))

    sentences_by_topic = {}
    for sent, gold_topic in unique_sentences:
        if eval_method == "t2f":
            idx = np.where(t2f_model.documents == sent)[0][0]
            topic = t2f_model.doc_topic_facet[idx]["topic"]
        elif eval_method == "gold_topics":
            topic = gold_topic
        elif eval_method == "no_topic_model":
            topic = 0
        if topic not in sentences_by_topic:
            sentences_by_topic[topic] = []
        sentences_by_topic[topic].append(sent)

    agg_cls = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                      linkage="average", distance_threshold=threshold)

    clusters = {}
    for topic in sentences_by_topic:
        clusters[topic] = {}
        docs = sentences_by_topic[topic]
        if len(docs) == 1:
            clusters[topic][0] = [docs[0]]
            continue
        t_doc_idxs = [np.where(t2f_model.documents == doc)[0][0] for doc in docs]
        dist_mat = 1 - cosine_similarity(t2f_model.document_vectors[t_doc_idxs])
        clustering = agg_cls.fit(dist_mat)
        for cluster in clustering.labels_:
            c_doc_idxs = np.where(clustering.labels_ == cluster)[0]
            clusters[topic][cluster] = [doc for idx, doc in enumerate(docs)
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


def best_clustering_split(t2f_model, eval_method, split, test_path_tplt):
    dev_file = test_path_tplt.format(split=split, mode="dev")
    test_file = test_path_tplt.format(split=split, mode="test")

    best_f1 = 0
    best_threshold = 0

    for threshold_int in range(0, 20):
        threshold = threshold_int / 20
        clusters = hclustering(t2f_model, dev_file, eval_method, threshold)
        __, __, f1_mean = eval_split(clusters, dev_file)

        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_threshold = threshold

    # print("Best threshold on dev:", best_threshold)

    # Compute clusters on test
    clusters = hclustering(t2f_model, test_file, eval_method, best_threshold)
    return clusters


def eval_t2f_hcl(t2f_model, eval_method, project_path):
    test_path_tplt = os.path.join(project_path, "datasets", "ukp_aspect", "splits",
                                  "{split}", "{mode}.tsv")

    all_f1_sim = []
    all_f1_dissim = []
    all_f1 = []
    for split in [0, 1, 2, 3]:
        # print("\n==================")
        # print("Split:", split)
        test_file = test_path_tplt.format(split=split, mode="test")
        clusters = best_clustering_split(t2f_model, eval_method, split,
                                         test_path_tplt)
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


def eval_t2f_full(t2f_model, project_path):
    test_file = os.path.join(project_path, "datasets", "ukp_aspect", "splits",
                             "all_data.tsv")
    test_data = []
    with open(test_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar=None)
        for splits in csvreader:
            splits = map(str.strip, splits)
            __, sentence_a, sentence_b, label = splits
            label_bin = 1 if label in ['SS', 'HS'] else 0
            test_data.append((sentence_a, sentence_b, label_bin))

    y_true = np.zeros(len(test_data))
    y_pred = np.zeros(len(test_data))

    for idx, row in enumerate(test_data):
        sentence_a, sentence_b, label_bin = row
        if label_bin == 1:
            y_true[idx] = 1

        idx_a = np.where(t2f_model.documents == sentence_a)[0][0]
        idx_b = np.where(t2f_model.documents == sentence_b)[0][0]

        topic_a = t2f_model.doc_topic_facet[idx_a]["topic"]
        topic_b = t2f_model.doc_topic_facet[idx_b]["topic"]
        facet_a = t2f_model.doc_topic_facet[idx_a]["facet"]
        facet_b = t2f_model.doc_topic_facet[idx_b]["facet"]

        if topic_a == topic_b and facet_a == facet_b:
            y_pred[idx] = 1

    f_sim = f1_score(y_true, y_pred, pos_label=1)
    f_dissim = f1_score(y_true, y_pred, pos_label=0)
    f_mean = np.mean([f_sim, f_dissim])
    print("F-Mean: %.4f" % f_mean)
    print("F-sim: %.4f" % f_sim)
    print("F-dissim: %.4f" % f_dissim)
