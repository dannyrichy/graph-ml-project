"""
Common utility functions
"""
import csv
import json
import pickle
from operator import methodcaller

import numpy as np
from google.colab import files
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate


# store all data from node classification
def store_node_classify_results(results, embeddings, labels, dataset, model):
    write_object(results, f"{dataset}_{model}_results.pickle")
    write_object(embeddings, f"{dataset}_{model}_embeddings.pickle")
    write_object(labels, f"{dataset}_{model}_labels.pickle")


# write oject into pickle file
def write_object(obj, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    files.download(filename)


# function to get labels of embedding based on nodelist order
def get_labels(nodelist, labels_dict):
    y = [labels_dict[x] for x in nodelist]
    return y


# read edges: Blog Catalog, Flickr, Youtube; (" ", 2) - Cora, Epinion, Twiiter
def read_edges(filename, separator=",", header=0):
    with open(filename, 'r') as f:
        edge_list = list(map(lambda x: (x[0], x[1], 1), list(map(methodcaller("split", separator), f.read().splitlines()[header:]))))
    return edge_list


# read labels from soc files: Blog Catalog, Flickr, Youtube
def read_labels(filename):
    with open(filename, 'r') as f:
        labels = dict(map(lambda x: (x[0], x[1]), list(map(methodcaller("split", ","), f.read().splitlines()))))
    return labels


# read labels from Cora
def read_cora_labels(filename):
    with open(filename, 'r') as f:
        labels = dict([(str(i + 1), val) for i, val in enumerate(f.read().splitlines())])
    return labels


# read edges from Pub Med
def read_pub_med_edges(filename, header=2):
    with open(filename, 'r') as f:
        edge_list = list(map(lambda x: (x[1][6:], x[3][6:], 1), list(map(methodcaller("split", '\t'), f.read().splitlines()[header:]))))
    return edge_list


# read labels from Pub Med
def read_pub_med_labels(filename, header=2):
    with open(filename, 'r') as f:
        labels = dict(map(lambda x: (x[0], x[1][6]), list(map(methodcaller("split", '\t'), f.read().splitlines()[header:]))))
    return labels


# read edges facebook
def read_facebook_edges(filename):
    with open(filename, 'r') as f:
        edge_list = list(map(lambda x: (x[0], x[1], 1), csv.reader(f)))[1:]
    return edge_list


# read labels facebook
def read_facebook_labels(filename):
    with open(filename, 'r', encoding="utf8") as f:
        labels = dict([(x[0], x[3]) for x in csv.reader(f)][1:])
    return labels


# read edges from Reddit
def read_reddit_edges(filename):
    with open(filename, 'rb') as f:
        edge_list = pickle.load(f)
    edge_list = [(u, v, 1) for u, v in edge_list]
    return edge_list


# read labels from Reddit
def read_reddit_labels(filename):
    with open(filename, 'r') as f:
        labels = json.load(f)
    return labels


# Logistic Regression - Node Classifer
def node_classifier(x, y):
    classifier = LogisticRegression(multi_class='ovr', solver='sag', n_jobs=-1, random_state=42)
    cv = cross_validate(classifier, x, y, scoring=('f1_micro', 'f1_macro'))
    print(cv)
    print("RESULTS:\nF1 Micro:", cv['test_f1_micro'].mean(), "\nF1 Macro:", cv['test_f1_macro'].mean())
    return cv


def d_link_pred(G, edges, labels, p=1):
    '''
    edges: array([node, node], ....)

    labels:  array(0, 1, 0, 1.....)
    '''

    mask = labels == 0
    neg_labels = labels[mask]

    mask_inv = labels == 1
    pos_labels = labels[mask_inv]
    pos_edges = edges[mask_inv]

    num_change = int(len(neg_labels) * p)

    new_edges = edges.copy()
    # count = 0
    count_pos = 0
    indices = np.where(labels == 0)

    for i in indices[0][:num_change]:
        while True:
            temp = np.array([pos_edges[count_pos][1], pos_edges[count_pos][0]])
            # print(temp)
            ch = (temp[0], temp[1])
            if not G.has_edge(*ch):
                new_edges[i] = temp
                count_pos += 1
                break
            else:
                count_pos += 1
    return new_edges, labels


# Link prediction(y_pred, y_true):
def link_prediction(y_pred, y_true):
    """

    :param y_pred:
    :type y_pred: numpy.ndarray

    :param y_true:
    :type y_true: numpy.ndarray

    :return:
    :rtype: float
    """
    y_pred[y_pred >= 0.0] = 1
    y_pred[y_pred < 0.0] = 0
    return roc_auc_score(y_true=y_true, y_score=y_pred)


class AliasTable:
    def __init__(self, prob_dist):
        """
        Class to generate the alias table

        :param prob_dist: Probability distribution to use
        :type prob_dist: list

        :return: None
        :rtype: Nothing
        """
        self.prob = prob_dist
        self.num_pts = len(self.prob)
        self.accept = np.zeros(self.num_pts)
        self.alias = np.zeros(self.num_pts)
        self.create_alias_table()

    def create_alias_table(self):
        """
        Generates the alias and accept list
        :return: Nothing
        :rtype: None
        """
        small, large = list(), list()
        area_ratio_ = np.array(self.prob) * self.num_pts
        for i, prob in enumerate(area_ratio_):
            if prob < 1.0:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            small_idx, large_idx = small.pop(), large.pop()
            self.accept[small_idx] = area_ratio_[small_idx]
            self.alias[small_idx] = large_idx
            area_ratio_[large_idx] = area_ratio_[large_idx] - (1 - area_ratio_[small_idx])
            if area_ratio_[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)

        while large:
            large_idx = large.pop()
            self.accept[large_idx] = 1
        while small:
            small_idx = small.pop()
            self.accept[small_idx] = 1

    def alias_sample(self):
        """
        Sample from the generated list

        :return: index
        :rtype: int
        """
        i = int(np.random.random() * self.num_pts)
        return i if np.random.random() < self.accept[i] else self.alias[i]
