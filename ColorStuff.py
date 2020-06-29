from __future__ import unicode_literals
import spacy
import numpy as np
from numpy import dot
from numpy.linalg import norm


nlp = spacy.load('en_core_web_md')

def vec(s):
    return nlp.vocab[s].vector

nlp.max_length = 1555172

doc = nlp(open("pg17.txt").read())

tokens = list(set([w.text for w in doc if w.is_alpha]))

def cosine(v1, v2):
    if norm(v1) > 0 and norm(v2) > 0:
        return dot(v1, v2) / (norm(v1) * norm(v2))
    else:
        return 0.0

def meanv(coords):
    # assumes every item in coords has same length as item 0
    sumv = [0] * len(coords[0])
    for item in coords:
        for i in range(len(item)):
            sumv[i] += item[i]
    mean = [0] * len(sumv)
    for i in range(len(sumv)):
        mean[i] = float(sumv[i]) / len(coords)
    return mean

def spacy_closest(token_list, vec_to_check, n=10):
    return sorted(token_list,
                  key=lambda x: cosine(vec_to_check, vec(x)),
                  reverse=True)[:n]


print(spacy_closest(tokens, vec("Christ")))








