#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : lda_cluster.py
# Create date : 2019-08-15 17:25
# Modified date : 2019-08-15 22:04
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import os,sys
from gensim.models import LdaModel
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim import similarities
from gensim import corpora

def create_data(corpus_path):
    '''构建数据，先后使用doc2bow和tfidf model对文本进行向量表示'''
    sentences = []
    target_lt = []
    sentence_dict={}
    count=0
    for line in open(corpus_path):
        line = line.strip().split('\t')
        if len(line) == 2:
            sentence_dict[count]=line[1]
            category=line[0]
            target_lt.append(category)
            count+=1
            sentences.append(line[1].split(' '))
        else:
            print(line)
            #break

    print(count)
    #对文本进行处理，得到文本集合中的词表
    dictionary = corpora.Dictionary(sentences)
    #利用词表，对文本进行cbow表示
    corpus = [dictionary.doc2bow(text) for text in sentences]
    print(len(corpus))
    
    #利用cbow，对文本进行tfidf表示
    tfidf=TfidfModel(corpus)
    corpus_tfidf=tfidf[corpus]
    return sentence_dict, dictionary, corpus, corpus_tfidf, target_lt

def lda_model(sentence_dict, dictionary, corpus, corpus_tfidf, cluster_keyword_lda, target_lt, num_cluster=11):
    '''使用lda模型，获取主题分布'''
    lda = LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=num_cluster)
    f_keyword = open(cluster_keyword_lda, 'w+')
    for topic in lda.print_topics(num_cluster, 53):
        #print('***************************')
        words=[]
        for word in topic[1].split('+'):
            word = word.split('*')[1].replace(' ','')
            words.append(word)
        f_keyword.write(str(topic[0])+'\t'+','.join(words)+'\n')

    #利用lda模型，对文本进行向量表示，这相当于与tfidf文档向量表示进行了降维，维度大小是设定的主题数目  
    corpus_lda = lda[corpus_tfidf]
    write_results("./results_lda.txt", corpus_lda, target_lt)

    return lda

def write_results(file_name, corpus, target_lt):
    f_docs=open(file_name,'w+')
    count=0

    for doc in corpus:
        max_index = 0
        max_value = 0.0
        for i in range(len(doc)):
            if doc[i][1] > max_value:
                max_value = doc[i][1]
                max_index = doc[i][0]
        f_docs.write(str(str(target_lt[count]))+','+str(max_index)+'\n')
        count+=1
    f_docs.close()


def lsi_model(sentence_dict,dictionary,corpus,corpus_tfidf,cluster_keyword_lsi, target_lt, num_cluster=11):
    '''使用lsi模型，获取主题分布'''
    lsi = LsiModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=num_cluster)
    f_keyword = open(cluster_keyword_lsi, 'w+')
    for topic in lsi.print_topics(num_cluster, 50):
        words=[]
        for word in topic[1].split('+'):
            word=word.split('*')[1].replace(' ','')
            words.append(word)
        f_keyword.write(str(topic[0])+'\t'+','.join(words)+'\n')
   
    corpus_lsi = lsi[corpus_tfidf]
    write_results("./results_lsi.txt", corpus_lsi, target_lt)
    return lsi

if __name__=="__main__":
    corpus_path = "./data/corpus_train.txt"
    cluster_keyword_lda = './cluster_keywords_lda.txt'
    cluster_keyword_lsi = './cluster_keywords_lsi.txt'
    sentence_dict, dictionary, corpus, corpus_tfidf, target_lt = create_data(corpus_path)
    lsi_model(sentence_dict, dictionary, corpus, corpus_tfidf, cluster_keyword_lsi, target_lt)
    lda_model(sentence_dict, dictionary, corpus, corpus_tfidf, cluster_keyword_lda, target_lt)

