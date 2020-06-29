from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import re
from bs4 import BeautifulSoup
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text


BRCA1_file = open("/Users/jameswengler/PycharmProjects/WordEmbedding/BRCA-Article1.txt")
BRCA2_file = open("/Users/jameswengler/PycharmProjects/WordEmbedding/BRCA-Article2.txt")
BRCA3_file = open("/Users/jameswengler/PycharmProjects/WordEmbedding/BRCA-Article3.txt")
Zebra1_file = open("/Users/jameswengler/PycharmProjects/WordEmbedding/ZebraFish-Article1.txt")
Zebra2_file = open("/Users/jameswengler/PycharmProjects/WordEmbedding/Zebra-Artvile2.txt")

BRCA1_text = BRCA1_file.read()
BRCA1_text = cleanText(BRCA1_text)

BRCA2_text = BRCA2_file.read()
BRCA2_text = cleanText(BRCA2_text)

BRCA3_text = BRCA3_file.read()
BRCA3_text = cleanText(BRCA3_text)

Zebra_text1 =Zebra1_file.read()
Zebra_text1 = cleanText(Zebra_text1)

Zebra_text2 =Zebra2_file.read()
Zebra_text2 = cleanText(Zebra_text2)


data = [BRCA2_text, BRCA3_text]

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

max_epochs = 100
vec_size = 200
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")

from gensim.models.doc2vec import Doc2Vec

model= Doc2Vec.load("d2v.model")
#to find the vector of a document which is not in training data
test_data = word_tokenize(Zebra_text2.lower())
v1 = model.infer_vector(test_data)

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar('1')
print(similar_doc)

# test_data2 = word_tokenize(BRCA3_text.lower())
# v2 = model.infer_vector(test_data2)
#
# # to find most similar doc using tags
# similar_doc = model.docvecs.most_similar('1')
# print(similar_doc)


# test_data = word_tokenize(Zebra_text2.lower())
# v1 = model.infer_vector(test_data)
#
# # to find most similar doc using tags
# similar_doc = model.docvecs.most_similar('1')
# print(similar_doc)
# #
# # to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
# print(model.docvecs['1'])


#BRCA1_file = open("/Users/jameswengler/PycharmProjects/WordEmbedding/BRCA-Article1.txt")
#BRCA2_file = open("/Users/jameswengler/PycharmProjects/WordEmbedding/BRCA-Article2.txt")
#BRCA3_file = open("/Users/jameswengler/PycharmProjects/WordEmbedding/BRCA-Article3.txt")
#Zebra1_file = open("/Users/jameswengler/PycharmProjects/WordEmbedding/ZebraFish-Article1.txt")

#BRCA1_tagged = TaggedDocument(BRCA1_file, tags = ["BRCA_PAPER"])
#BRCA2_tagged = TaggedDocument(BRCA2_file, tags = ["BRCA_PAPER"])
#BRCA3_tagged = TaggedDocument(BRCA3_file, tags = ["BRCA_PAPER"])
#Zebra_tagged = TaggedDocument(Zebra1_file, tags = ["NOT_BRCA_PAPER"])

#documents = [BRCA1_tagged, BRCA2_tagged, BRCA3_tagged, Zebra_tagged]

#model = Doc2Vec(documents, vector_size=4, window=2, min_count=1, workers=4)





