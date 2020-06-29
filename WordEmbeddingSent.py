from __future__ import unicode_literals
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine
import numpy as np

def vec(s):
    return nlp.vocab[s].vector


def getSentenceSimilarity(sent1, sent2):
    word1 = removeStopWords(sent1)
    word2 = removeStopWords(sent2)
    # token1 = tokenize(word1)
    # token2 = tokenize(word2)

    first_word = word1[0]
    first_vec = vec(first_word)

    for i, word in enumerate(word1):
        if(i != 0):
            new_vec = vec(word)
            if new_vec is not None:
                first_vec = np.add(first_vec, new_vec)

    # Red = [0,1,-2,3]
    # dog = [2,-2,2,1]
    # red dog = [2,3,4,4]

    new_first_word = word2[0]
    new_first_vec = vec(new_first_word)

    for i, word in enumerate(word2):
        if (i != 0):
            new_vec = vec(word)
            if new_vec is not None:
                new_first_vec = np.add(new_first_vec, new_vec)

    cosine_similarity = 1 - cosine(first_vec, new_first_vec)
    return cosine_similarity


def tokenize(words):
    sep = ' '
    token_str = sep.join(words)
    tokens = nlp(token_str)
    return tokens

def removeStopWords(sentence):
    stop_words = set(stopwords.words('english'))

    words = word_tokenize(sentence)

    filtered_sentence = []

    for w in words:
        if w not in stop_words:
            filtered_sentence.append(w)

    return filtered_sentence

nlp = spacy.load('en_core_web_lg')
nlp.max_length = 1555172

sent1 = "Association of BRCA Mutation Types, Imaging Features, and Pathologic Findings in Patients With Breast Cancer With BRCA1 and BRCA2 Mutations"
sent2 = "Best Teaching Practices in Anatomy Education: A Critical Review"


print(getSentenceSimilarity(sent1, sent2))







