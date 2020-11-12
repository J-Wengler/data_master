from nltk.tokenize import sent_tokenize, word_tokenize
import multiprocessing
from gensim.models import Word2Vec
import gensim.models
from gensim import utils
from gensim.models import KeyedVectors

all_text = ""
with open('Models/starGEO.txt', 'r+') as in_file:
          all_text = in_file.read()


parsed_text = []

sentences = sent_tokenize(all_text)
for sent in sentences:
    words = word_tokenize(sent)
    parsed_text.append(words)

model = Word2Vec(parsed_text, min_count = 3, size = 300, workers = multiprocessing.cpu_count(), window = 5, iter = 30, sg = 0)

model.save('/Models/customSpacyCBOW.bin')

model = KeyedVectors.load_word2vec_format('customSpacyCBOW.bin', binary=True)

#model = gensim.models.KeyedVectors.load_word2vec_format('/Models/customSpacy', binary=True)

#model = Word2Vec.load('/Models/customSpacy.model')

#print(model.most_similar(positive = ['BRCA'], topn=5))
