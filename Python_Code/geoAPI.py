import requests
import pandas as pd
import fasttext
import re
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

def cleanText(text):
    stemmer = WordNetLemmatizer()
    en_stop = set(nltk.corpus.stopwords.words('english'))
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
    tokens = text.split()
    #tokens = [stemmer.lemmatize(word) for word in tokens]
    #tokens = [word for word in tokens if word not in en_stop]
    #tokens = [word for word in tokens if len(word) > 3]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def getDocEmbedding(text, model):

    #text = cleanText(text)

    tr4w = TextRank4Keyword()
    tr4w.analyze(text, candidate_pos = ['NOUN', 'PROPN','ADJ'], window_size=6, lower=False)
    ans = tr4w.get_keywords(10)
    if ans is not None:
        allWords = []
        for key in ans:
            allWords.append(key)
        first_word = allWords[0]
        doc_vec = model[first_word]
        numWords = 1
        for i, word in enumerate(allWords):
            if (i != 0):
                new_vec = model[word]
                if new_vec is not None:
                    numWords += 1
                    doc_vec = np.add(doc_vec, new_vec)
        doc_vec = doc_vec / numWords
        return doc_vec
    else:
        allWords = word_tokenize(text)
        first_word = allWords[0]
        doc_vec = model[first_word]
        numWords = 1
        for i, word in enumerate(allWords):
            if (i != 0):
                new_vec = model[word]
                if new_vec is not None:
                    numWords += 1
                    doc_vec = np.add(doc_vec, new_vec)
        doc_vec = doc_vec / numWords
        return doc_vec
# Fetch first 10 series, defaults to 100
#r = requests.get('http://stargeo.org/api/v2/series/?limit=10')
#assert r.ok
#data = r.json()

series_to_summary = {}

#data = pd.read_json('http://stargeo.org/api/v2/series/?limit=100000')

#json_file = requests.get('http://stargeo.org/api/v2/search/?q=BRCA+and+CANCER').json()

#print(requests.get('http://stargeo.org/api/v2/search/?q=BRCA+and+CANCER').status_code)
rq = requests.get('http://stargeo.org/api/v2/series/?limit=1000000').json
data = pd.read_json(rq)

for row in data['results']:
    temp_dict = row['attrs']
    name = row['gse_name']
    summary = temp_dict['summary']
    title = temp_dict['title']
    series_to_summary[name] = [summary,title]


