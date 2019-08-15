#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : main.py
# Create date : 2019-08-15 16:47
# Modified date : 2019-08-15 22:06
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

from lda_cluster import lsi_model
from lda_cluster import lda_model
from lda_cluster import create_data

from kmeans_cluster import  tfidf_vector
from kmeans_cluster import  best_kmeans
from kmeans_cluster import  cluster_kmeans

def run_lda_cluster():
    corpus_path = "./data/corpus_train.txt"
    cluster_keyword_lda = './cluster_keywords_lda.txt'
    cluster_keyword_lsi = './cluster_keywords_lsi.txt'
    sentence_dict, dictionary, corpus, corpus_tfidf, target_lt = create_data(corpus_path)
    num_clusters = 7
    lsi = lsi_model(sentence_dict,dictionary,corpus,corpus_tfidf,cluster_keyword_lsi, target_lt, num_clusters)
    lda = lda_model(sentence_dict, dictionary, corpus, corpus_tfidf,cluster_keyword_lda, target_lt, num_clusters)

def run_kmeans_cluster():
    corpus_train = "./data/corpus_train.txt"
    cluster_docs = "./cluster_result_document.txt"
    cluster_keywords = "./cluster_result_keyword.txt"
    num_clusters = 7
    tfidf_train,word_dict = tfidf_vector(corpus_train)

    best_kmeans(tfidf_train,word_dict)
    cluster_kmeans(tfidf_train, word_dict, cluster_docs, cluster_keywords, num_clusters)

def run():
    run_lda_cluster()
    run_kmeans_cluster()

run()
