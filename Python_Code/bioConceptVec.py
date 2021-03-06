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
from gensim.models import KeyedVectors
import os, sys, json, numpy as np

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
    tokens = [stemmer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in en_stop]
    tokens = [word for word in tokens if len(word) > 3]

    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def getDocEmbedding(pathToDoc, model):

    in_file = open(pathToDoc)
    in_text = in_file.read()
    in_text = cleanText(in_text)

    allWords = word_tokenize(in_text)

    first_word = allWords[0]
    doc_vec = np.zeros((100,))
    #doc_vec = model[first_word]
    numWords = 0
    for i, word in enumerate(allWords):
        if word in model:
            new_vec = model[word]
            if new_vec is not None:
                numWords = i
                doc_vec = np.add(doc_vec, new_vec)

    doc_vec = doc_vec / numWords
    return doc_vec

def cleanDataTrainModel(pathToTextFile):
    in_file = open(pathToTextFile)
    in_text = in_file.read()
    in_text = cleanText(in_text)
    out_file = open('/Users/jameswengler/PycharmProjects/WordEmbedding/cleanAllText.txt', 'w+')
    out_file.write(in_text)
    model = fasttext.train_unsupervised('/Users/jameswengler/PycharmProjects/WordEmbedding/cleanAllText.txt', model='skipgram')
    os.remove('/Users/jameswengler/PycharmProjects/WordEmbedding/cleanAllText.txt')
    return model

def trainModel(pathToCleanFile):
    model = fasttext.train_unsupervised(pathToCleanFile, model='skipgram')
    return model



def load_embedding(path):
    embedding = KeyedVectors.load_word2vec_format(path, binary = True)
    print('embedding loaded from', path)
    return embedding

#model = load_embedding("/BioWordModel/concept_model.bin")


#model = trainModel('/Users/jameswengler/PycharmProjects/WordEmbedding/wikiArticles.txt')
#model = fasttext.load_model('/BioWordModel/model.bin')
#allDocs = []
#for i in range(1,11):
#    inString = '/TitleSummary{}.txt'.format(i)
#    temp_vec = getDocEmbedding(inString, model)
#    allDocs.append(temp_vec)



#with open("/BioWordModel/concept_model.json") as json_file:  
#    concept_vectors = json.load(json_file)
#print('load', len(concept_vectors), 'concepts')

pathToDoc='/TitleSummary1.txt'
in_file = open(pathToDoc)
in_text = in_file.read()
in_text = cleanText(in_text)
allWords = word_tokenize(in_text)
taggedWords = nltk.pos_tag(allWords)
for word in taggedWords:
    print(word[1])


# data = StandardScaler().fit_transform(allDocs)
#firstDoc = 1
#secondDoc = 1
# out_file = open("output/FastText-AbstractAndTitle.txt", 'w+')
#for doc in allDocs:
#    secondDoc = 1
#    for doc2 in allDocs:
#        cosine_similarity = 1 - cosine(doc, doc2)
#        cosine_similarity *= 100
#        print("Document {} is {}% similar to Document {}".format(firstDoc, cosine_similarity, secondDoc))
#        # out_file.write("Document {} is {}% similar to Document {}".format(firstDoc, cosine_similarity, secondDoc))
#        # out_file.write('\n')
#        secondDoc += 1
#    # out_file.write('\n')
#    print()
#    firstDoc += 1



#pca = PCA(n_components=2)

#principalComponents = pca.fit_transform(allDocs)

#article = 1
#for coor in principalComponents:
#    if(article == 1 or article == 2 or article == 3):
#        plt.scatter(x=coor[0], y=coor[1], c = 'b', s=300)
#    elif(article == 4 or article == 5):
#        plt.scatter(x=coor[0], y=coor[1], c='r', s=300)
#    elif(article == 6):
#        plt.scatter(x=coor[0], y=coor[1], c='g', s=300)
#    elif(article == 7):
#        plt.scatter(x=coor[0], y=coor[1], c='c', s=300)
#    elif(article == 8):
#        plt.scatter(x=coor[0], y=coor[1], c='m', s=300)
#    elif(article == 9):
#        plt.scatter(x=coor[0], y=coor[1], c='y', s=300)
#    else:
#        plt.scatter(x=coor[0], y=coor[1], c='k', s=300)
#
#    article += 1


# plt.show()
#plt.savefig("images/FastText-TitleAndAbstract.png")
