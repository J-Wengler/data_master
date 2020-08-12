import pke
import fasttext
from bs4 import BeautifulSoup
import os
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
import io
import wikipedia
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt
from TextRank import TextRank4Keyword
import pke
import os
import requests
from gensim.models import KeyedVectors
import pandas as pd
import re

def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'^b\s+', '', text)
    text = text.lower()
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    text = text.replace(',', ' ')
    text = re.sub('\n', ' ', text)
    text = re.sub('[n|N]o\.', 'number', text)

    return text

def getAllArticles():
    rq = requests.get('http://stargeo.org/api/v2/series/?limit=1000000').json()
    #data = pd.read_json(rq)
    series_to_summary = {}
    for row in rq['results']:
        temp_dict = row['attrs']
        name = row['gse_name']
        summary = temp_dict['summary']
        title = temp_dict['title']
        full_text = summary + title
        full_text = cleanText(full_text)
        series_to_summary[name] = full_text

    return series_to_summary

def getKeywordEmbedding(keywords,model, numWords):
    doc_vec = np.zeros((100,))
    for i, word in enumerate(keywords):
        if type(word) is tuple:
            word = word[0]
        #print("GOOD THINGS ARE HAPPENING")
        #print("CURRENT WORD : {}".format(word))
        word_list = word.split()
        if len(word_list) > 1:
            new_vec = getKeywordEmbedding(word_list, model, numWords)
            if new_vec is not None:
                numWords += 1
                doc_vec = np.add(doc_vec, new_vec)
        else:
            error = False
            try:
                new_vec = model[word_list[0]]
            except KeyError or ValueError:
                #print("{} is not found in the model... Skipping".format(word_list[0]))
                error = True
                break
            if new_vec is not None and error is False:
                numWords += 1
                doc_vec = np.add(doc_vec, new_vec)

    if (numWords != 0):
        doc_vec = doc_vec / numWords
    return doc_vec

def load_embedding(path):
    embedding = KeyedVectors.load_word2vec_format(path, binary = True)
    print('embedding loaded from', path)
    return embedding    

def getNamesToQuery(path):
    filePath = path + 'names_to_query.txt'
    files = []
    with open(filePath, 'r+') as in_file:
        for line in in_file:
            names = line.split(' ')
            for name in names:
                files.append(name.upper().strip())
    return files


def getTopicRank(model, articles):
    path = '/Models/Queries/q1/'
    filenames = getNamesToQuery(path)
    embeddings = []
    sim_to_name = {}
    for filename in filenames:
        method = pke.unsupervised.TopicRank()
        method.load_document(input='/Models/Queries/q1/{}.txt'.format(filename), language='en')
        method.candidate_selection()
        method.candidate_weighting()
        keyphrases = method.get_n_best(n=5)
        embeddings.append(getKeywordEmbedding(keyphrases,model, 0))
    print("LENGTH --- {}".format(len(articles)))
    num_articles = 0
    for art in articles:
        num_articles += 1
        print("NUM --- {}".format(num_articles))
        method = pke.unsupervised.TopicRank()
        method.load_document(input=articles[art], language='en')
        method.candidate_selection()
        error = False
        try:
            method.candidate_weighting()
        except ValueError:
            print("Method was unable to find keyphrases for identification. Setting similarity to 0%")
            sim_to_name[art] = 0
            error = True
        if (error is not True):
            keyphrases = method.get_n_best(n=5)
            cur_vec = getKeywordEmbedding(keyphrases, model, 0)
            avg_sim = 0
            num_embeddings = 0
            for f in embeddings:
                sim = 1 - cosine(cur_vec, f)
                avg_sim += sim
                num_embeddings += 1
            sim_to_name[art] = avg_sim / num_embeddings
    for name in sim_to_name:
        print("{} ---- {}".format(name, sim_to_name[name]))



#def getTFIDF(model):
#    path = '/Models/Queries/q1/'
#    for filename in os.listdir(path):
#        method = pke.unsupervised.TfIdf()
#        if filename != 'names.txt':
#            method.load_document(input='/Models/Queries/q1/{}'.format(filename), language='en')
#            method.candidate_selection()
#            method.candidate_weighting()
#            keyphrases = method.get_n_best(n=5)
#            print(getKeywordEmbedding(keyphrases,model, 0))
#
#def getKPMiner(model):
#    path = '/Models/Queries/q1/'
#    for filename in os.listdir(path):
#        method = pke.unsupervised.KPMiner()
#        if filename != 'names.txt':
#            method.load_document(input='/Models/Queries/q1/{}'.format(filename), language='en')
#            method.candidate_selection()
#            method.candidate_weighting()
#            keyphrases = method.get_n_best(n=5)
#            print(getKeywordEmbedding(keyphrases,model, 0))
#
#def getYAKE(model):
#    path = '/Models/Queries/q1/'
#    for filename in os.listdir(path):
#        method = pke.unsupervised.TopicRank()
#        if filename != 'names.txt':
#            method.load_document(input='/Models/Queries/q1/{}'.format(filename), language='en')
#            method.candidate_selection()
#            method.candidate_weighting()
#            keyphrases = method.get_n_best(n=5)
#            print(getKeywordEmbedding(keyphrases,model, 0))
#
#def getTextRank(model):
#    path = '/Models/Queries/q1/'
#    for filename in os.listdir(path):
#        method = pke.unsupervised.YAKE()
#        if filename != 'names.txt':
#            method.load_document(input='/Models/Queries/q1/{}'.format(filename), language='en')
#            method.candidate_selection()
#            method.candidate_weighting()
#            keyphrases = method.get_n_best(n=5)
#            print(getKeywordEmbedding(keyphrases,model, 0))
#
#def getSingleRank(model):
#    path = '/Models/Queries/q1/'
#    for filename in os.listdir(path):
#        method = pke.unsupervised.SingleRank()
#        if filename != 'names.txt':
#            method.load_document(input='/Models/Queries/q1/{}'.format(filename), language='en')
#            method.candidate_selection()
#            method.candidate_weighting()
#            keyphrases = method.get_n_best(n=5)
#            print(getKeywordEmbedding(keyphrases,model, 0))
#
#def getTopicalRank(model):
#    path = '/Models/Queries/q1/'
#    for filename in os.listdir(path):
#        method = pke.unsupervised.TopicalPageRank()
#        if filename != 'names.txt':
#            method.load_document(input='/Models/Queries/q1/{}'.format(filename), language='en')
#            method.candidate_selection()
#            method.candidate_weighting()
#            keyphrases = method.get_n_best(n=5)
#            print(getKeywordEmbedding(keyphrases,model, 0))
#
#def getPositionRank(model):
#    path = '/Models/Queries/q1/'
#    for filename in os.listdir(path):
#        method = pke.unsupervised.PositionRank()
#        if filename != 'names.txt':
#            method.load_document(input='/Models/Queries/q1/{}'.format(filename), language='en')
#            method.candidate_selection()
#            method.candidate_weighting()
#            keyphrases = method.get_n_best(n=5)
#            print(getKeywordEmbedding(keyphrases,model, 0))
#
#def getMultipartiteRank(model):
#    path = '/Models/Queries/q1/'
#    for filename in os.listdir(path):
#        method = pke.unsupervised.MultipartiteRank()
#        if filename != 'names.txt':
#            method.load_document(input='/Models/Queries/q1/{}'.format(filename), language='en')
#            method.candidate_selection()
#            method.candidate_weighting()
#            keyphrases = method.get_n_best(n=5)
#            print(getKeywordEmbedding(keyphrases,model, 0))



all_articles = getAllArticles()
model = load_embedding("/Models/concept_model.bin")

getTopicRank(model, all_articles)
print("PASSED")

#getTFIDF(model)
#print("PASSED")
#getKPMiner(model)
#print("PASSED")
#getYAKE(model)
#print("PASSED")
#getTextRank(model)
#print("PASSED")
#getSingleRank(model)
#print("PASSED")
#getTopicalRank(model)
#print("PASSED")
#getPositionRank(model)
#print("PASSED")
#getMultipartiteRank(model)
#print("PASSED")
