"""
Evaluates the performance on the UKP ASPECT Corpus with hierachical clustering.

Greedy hierachical clustering.
Merges two clusters if the pairwise mean cluster similarity is larger than a threshold.
Merges clusters with highest similarity first
Uses dev set to determine the threshold for supervised systems
"""
import csv
import os
import joblib
import sys

import pandas as pd
import numpy as np
import scipy
import scipy.spatial.distance
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering


class PriorityQueue(object):
    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0

    # for inserting an element in the queue
    def insert(self, data):
        self.queue.append(data)

    # Removes all element addressing a cluster key
    def remove_clusters(self, cluster_key):
        i = 0
        while i < len(self.queue):
            ele = self.queue[i]
            if ele['cluster_a'] == cluster_key or ele['cluster_b'] == cluster_key:
                del self.queue[i]
            else:
                i += 1

    # for popping an element based on Priority
    def pop(self):
        max = 0
        for i in range(len(self.queue)):
            if self.queue[i]['cluster_sim'] > self.queue[max]['cluster_sim']:
                max = i
        item = self.queue[max]
        del self.queue[max]
        return item


class HierachicalClustering:
    """
    Simple clustering algorithm. Merges two clusters, if the cluster similarity is larger than the threshold.
    Highest similarities first.
    """
    def __init__(self, t2f_model, testfile, topic_model, np_mode=np.mean):
        self.t2f_model = t2f_model
        self.clusters = self.init_clusters(testfile, topic_model)
        self.np_mode = np_mode

    def init_clusters(self, testfile, topic_model):
        unique_sentences = set()
        with open(testfile, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter='\t', quotechar=None)
            for splits in csvreader:
                splits = map(str.strip, splits)
                topic, sentence_a, sentence_b, __ = splits
                unique_sentences.update([(sentence_a, topic), (sentence_b, topic)])
        clusters = {}
        for idx, row in enumerate(unique_sentences):
            sent, topic = row
            if topic_model:
                idx_t2f = np.where(self.t2f_model.documents == sent)[0][0]
                topic_t2f = self.t2f_model.doc_topic_facet[idx_t2f]["topic"]
                if topic_t2f not in clusters.keys():
                    clusters[topic_t2f] = {}
                clusters[topic_t2f][idx_t2f] = [sent]
            else:
                if topic not in clusters.keys():
                    clusters[topic] = {}
                clusters[topic][idx] = [sent]
        return clusters

    def compute_similarity_score(self, sentence_a, sentence_b):
        idx_a = np.where(self.t2f_model.documents == sentence_a)[0][0]
        idx_b = np.where(self.t2f_model.documents == sentence_b)[0][0]
        sim = 1 - scipy.spatial.distance.cosine(self.t2f_model.document_vectors[idx_a],
                                                self.t2f_model.document_vectors[idx_b])
        return sim

    def compute_cluster_sim(self, cluster_a, cluster_b):
        scores = []
        for sentence_a in cluster_a:
            for sentence_b in cluster_b:
                scores.append(self.compute_similarity_score(sentence_a, sentence_b))

        return self.np_mode(scores)

    def cluster_topics(self, threshold):
        for topic in self.clusters:
            topic_cluster = self.clusters[topic]
            self.run_clustering(topic_cluster, threshold)
        return self.clusters

    def run_clustering(self, clusters, threshold):
        queue = PriorityQueue()

        cluster_ids = list(clusters.keys())
        for i in range(0, len(cluster_ids)-1):
            for j in range(i+1, len(cluster_ids)):
                cluster_a = cluster_ids[i]
                cluster_b = cluster_ids[j]

                cluster_sim = self.compute_cluster_sim(clusters[cluster_a],
                                                       clusters[cluster_b])
                element = {'cluster_sim': cluster_sim, 'cluster_a': cluster_a,
                           'cluster_b': cluster_b}
                queue.insert(element)

        while not queue.isEmpty():
            element = queue.pop()
            if element['cluster_sim'] <= threshold:
                break

            self.merge_clusters(clusters, element['cluster_a'], element['cluster_b'])

            queue.remove_clusters(element['cluster_a'])
            queue.remove_clusters(element['cluster_b'])

            cluster_a = element['cluster_a']
            for cluster_b in clusters.keys():
                if cluster_a != cluster_b:
                    cluster_sim = self.compute_cluster_sim(clusters[cluster_a],
                                                           clusters[cluster_b])
                    element = {'cluster_sim': cluster_sim, 'cluster_a': cluster_a,
                               'cluster_b': cluster_b}
                    queue.insert(element)

    def merge_clusters(self, clusters, key_a, key_b):
        clusters[key_a] += clusters[key_b]
        del clusters[key_b]


# def get_clustering(t2f_model, testfile, topic_model, threshold):
#     cluster_alg = HierachicalClustering(t2f_model, testfile, topic_model)
#     clusters = cluster_alg.cluster_topics(threshold)
#     return clusters

def hclustering(t2f_model, testfile, use_topic_model, threshold):
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
        if use_topic_model:
            idx = np.where(t2f_model.documents == sent)[0][0]
            topic = t2f_model.doc_topic_facet[idx]["topic"]
        else:
            topic = gold_topic
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


def eval_split(clusters, labels_file, print_scores=False):
    test_data = []
    with open(labels_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar=None)
        for splits in csvreader:
            splits = map(str.strip, splits)
            __, sentence_a, sentence_b, label = splits
            label_bin = 1 if label in ['SS', 'HS'] else 0
            test_data.append((sentence_a, sentence_b, label_bin))

    sentences_cluster_id = {}
    for topic in clusters:
        topic_cluster = clusters[topic]
        for cluster_id in topic_cluster:
            for sentence in topic_cluster[cluster_id]:
                sentences_cluster_id[sentence] = str(topic) + "_" + str(cluster_id)

    y_true = np.zeros(len(test_data))
    y_pred = np.zeros(len(test_data))

    for idx, row in enumerate(test_data):
        sentence_a, sentence_b, label_bin = row
        if label_bin == 1:
            y_true[idx] = 1
        if sentences_cluster_id[sentence_a] == sentences_cluster_id[sentence_b]:
            y_pred[idx] = 1

    f_sim = f1_score(y_true, y_pred, pos_label=1)
    f_dissim = f1_score(y_true, y_pred, pos_label=0)
    f_mean = np.mean([f_sim, f_dissim])

    if print_scores:
        print("F-Sim: %.2f%%" % (f_sim * 100))
        print("F-Dissim: %.2f%%" % (f_dissim * 100))
        print("F-Mean: %.2f%%" % (f_mean * 100))
        acc = np.sum(y_true == y_pred) / len(y_true)
        print("Acc: %.2f%%" % (acc * 100))

    return f_sim, f_dissim, f_mean


def best_clustering_split(t2f_model, use_topic_model, split, test_path_tplt):
    dev_file = test_path_tplt.format(split=split, mode="dev")
    test_file = test_path_tplt.format(split=split, mode="test")

    best_f1 = 0
    best_threshold = 0

    for threshold_int in range(0, 20):
        threshold = threshold_int / 20
        clusters = hclustering(t2f_model, dev_file, use_topic_model, threshold)
        __, __, f1_mean = eval_split(clusters, dev_file)

        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_threshold = threshold

    # print("Best threshold on dev:", best_threshold)

    # Compute clusters on test
    clusters = hclustering(t2f_model, test_file, use_topic_model, best_threshold)
    return clusters


def eval_t2f_hcl(t2f_model, use_topic_model, project_path):
    test_path_tplt = os.path.join(project_path, "datasets", "ukp_aspect", "splits",
                                  "{split}", "{mode}.tsv")

    all_f1_sim = []
    all_f1_dissim = []
    all_f1 = []
    for split in [0, 1, 2, 3]:
        # print("\n==================")
        # print("Split:", split)
        test_file = test_path_tplt.format(split=split, mode="test")
        clusters = best_clustering_split(t2f_model, use_topic_model, split,
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

        # if y_true[idx] != y_pred[idx] and label in ['SS', 'HS']:
        #     print(topic, "\n", label, "\n", sentence_a, "\n", sentence_b, "\n",
        #           t2f_model.topic_words[topic_a][:5], "\n",
        #           t2f_model.topic_words[topic_b][:5], "\n",
        #           t2f_model.topic_facets[topic_a][facet_a]["words"][:5], "\n",
        #           t2f_model.topic_facets[topic_b][facet_b]["words"][:5], "\n",
        #           "\n\n")
    f_sim = f1_score(y_true, y_pred, pos_label=1)
    f_dissim = f1_score(y_true, y_pred, pos_label=0)
    f_mean = np.mean([f_sim, f_dissim])
    print("F-Mean: %.4f" % f_mean)
    print("F-sim: %.4f" % f_sim)
    print("F-dissim: %.4f" % f_dissim)


if __name__ == "__main__":
    ACL_PATH = ("/home/romain/projects/acl2019-BERT-argument-classification-"
                "and-clustering/argument-similarity")

    SOCSEMICS_PATH = "/home/romain/projects/socsemics"
    MODELS_DIR = os.path.join(SOCSEMICS_PATH, "experiments", "models")
    TM_DIR = os.path.join(MODELS_DIR, "topic_modelling", "st_tm")

    CLAIMIT_PATH = "/home/romain/projects/claimit"
    sys.path.append(CLAIMIT_PATH)

    model = joblib.load(os.path.join(TM_DIR, "ukp_corpus_nli_mpnet_base_v2"))
    eval_t2f_hcl(t2f_model=model,
                 topic_model=True,
                 project_path=ACL_PATH)
