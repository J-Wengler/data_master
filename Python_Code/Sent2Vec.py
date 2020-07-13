from __future__ import unicode_literals
from nltk.tokenize import sent_tokenize
import re
from bs4 import BeautifulSoup
import spacy
import scispacy
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

    total_acc = 0
    total_words = 0

    first_word = word1[0]
    first_vec = vec(first_word)

    for i, word in enumerate(word1):
        if(i != 0):
            new_vec = vec(word)
            if new_vec is not None:
                first_vec = np.add(first_vec, new_vec)

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

nlp = spacy.load("en_core_web_md")
# nlp = spacy.load('xx_ent_wiki_sm')
# nlp.max_length = 1555172
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    text = re.sub('\n', ' ', text)
    text = re.sub('[n|N]o\.', 'number', text)
    return text

Zebra1_file = open("/Users/jameswengler/PycharmProjects/WordEmbedding/Lawyer_Article.txt")
Zebra1_text = Zebra1_file.read()
Zebra1_text = cleanText(Zebra1_text)

sent_list1 = sent_tokenize(Zebra1_text)

BRCA2_file = open("/Users/jameswengler/PycharmProjects/WordEmbedding/BRCA-Article1.txt")
BRCA2_text = BRCA2_file.read()
BRCA2_text = cleanText(BRCA2_text)

sent_list2 = sent_tokenize(BRCA2_text)
#
# model = sent2vec.Sent2vecModel()
# model.load_model('model.bin')

TEST_SENT = "Skin Cancer Risk in BRCA1/2 Mutation Carriers"
#
# total_sim = 0
# num_sent = 0
# test_vec = model.embed_sentence(TEST_SENT)

# for sent in sent_list1:
#     temp_vec = model.embed_sentence(sent)
#     temp_sim = cosine_similarity(temp_vec, test_vec)
#     total_sim += temp_sim[0][0]
#     num_sent += 1
#
#
# avg_sim = total_sim / num_sent
# percent = round(avg_sim * 100 , 1)
# out = "\'" + TEST_SENT + "\' is " + str(percent) + "% similar to document 1"
# print(out)
#
# total_sim = 0
# num_sent = 0
#
# for sent in sent_list2:
#     temp_vec = model.embed_sentence(sent)
#     temp_sim = cosine_similarity(temp_vec, test_vec)
#     total_sim += temp_sim[0][0]
#     num_sent += 1
#
# avg_sim = total_sim / num_sent
# percent = round(avg_sim * 100 , 1)
# out = "\'" + TEST_SENT + "\' is " + str(percent) + "% similar to document 2"
# print(out)
#


total_sim = 0
num_sent = 0

for sent in sent_list1:
    temp_sim = getSentenceSimilarity(sent, TEST_SENT)
    total_sim += temp_sim
    num_sent += 1


avg_sim = total_sim / num_sent
percent = round(avg_sim * 100 , 1)
out = "\'" + TEST_SENT + "\' is " + str(percent) + "% similar to document 1"
print(out)

total_sim = 0
num_sent = 0

for sent in sent_list2:
    temp_sim = getSentenceSimilarity(sent, TEST_SENT)
    total_sim += temp_sim
    num_sent += 1

avg_sim = total_sim / num_sent
percent = round(avg_sim * 100 , 1)
out = "\'" + TEST_SENT + "\' is " + str(percent) + "% similar to document 2"
print(out)



# sent1 = "I like big dogs"
# sent2 = "The fact that the rat is sick is alarming"
# model = sent2vec.Sent2vecModel()
# model.load_model('model.bin')
# vec1 = model.embed_sentence(sent1)
# vec2 = model.embed_sentence(sent2)
#
# sim = cosine_similarity(vec1, vec2)
#
# percent = round(sim[0][0] * 100,1)
# out = "\'" + sent1 + "\' and \'" + sent2 + "\' are " + str(percent) + "% similar"
# print(out)
