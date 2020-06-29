from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import gensim
import re
from bs4 import BeautifulSoup
# def cleanText(text):
#     text = BeautifulSoup(text, "lxml").text
#     text = re.sub(r'\|\|\|', r' ', text)
#     text = re.sub(r'http\S+', r'<URL>', text)
#     text = text.lower()
#     text = text.replace('x', '')
#     return text
#
# BRCA1_file = open("/Users/jameswengler/PycharmProjects/WordEmbedding/BRCA-Article1.txt")
# BRCA2_file = open("/Users/jameswengler/PycharmProjects/WordEmbedding/BRCA-Article2.txt")
# BRCA3_file = open("/Users/jameswengler/PycharmProjects/WordEmbedding/BRCA-Article3.txt")
# Zebra1_file = open("/Users/jameswengler/PycharmProjects/WordEmbedding/ZebraFish-Article1.txt")
# Zebra2_file = open("/Users/jameswengler/PycharmProjects/WordEmbedding/Zebra-Artvile2.txt")
#
# BRCA1_text = BRCA1_file.read()
# BRCA1_text = cleanText(BRCA1_text)
#
# BRCA2_text = BRCA2_file.read()
# BRCA2_text = cleanText(BRCA2_text)
#
# BRCA3_text = BRCA3_file.read()
# BRCA3_text = cleanText(BRCA3_text)
#
# Zebra_text1 = Zebra1_file.read()
# Zebra_text1 = cleanText(Zebra_text1)
#
# Zebra_text2 = Zebra2_file.read()
# Zebra_text2 = cleanText(Zebra_text2)
#
file_docs = []

with open ('/Users/jameswengler/PycharmProjects/WordEmbedding/BRCA-Article1.txt') as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file_docs.append(line)

print("Number of documents:",len(file_docs))

gen_docs = [[w.lower() for w in word_tokenize(text)]
            for text in file_docs]

dictionary = gensim.corpora.Dictionary(gen_docs)


corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

tf_idf = gensim.models.TfidfModel(corpus)

sims = gensim.similarities.Similarity('workdir/',tf_idf[corpus],
                                        num_features=len(dictionary))

file2_docs = []

with open ('/Users/jameswengler/PycharmProjects/WordEmbedding/BRCA-Article2.txt') as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file2_docs.append(line)

print("Number of documents:",len(file2_docs))
for line in file2_docs:
    query_doc = [w.lower() for w in word_tokenize(line)]
    query_doc_bow = dictionary.doc2bow(query_doc)


# perform a similarity query against the corpus
query_doc_tf_idf = tf_idf[query_doc_bow]
# print(document_number, document_similarity)
print('Comparing Result:', sims[query_doc_tf_idf])