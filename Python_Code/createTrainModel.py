import gensim
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from bs4 import BeautifulSoup
import re
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    text = re.sub('\n', ' ', text)
    text = re.sub('[n|N]o\.', 'number', text)
    return text


input_file = open("/Users/jameswengler/PycharmProjects/WordEmbedding/allText.txt")
input_text = input_file.read()
input_text = cleanText(input_text)

#  The dog is blue
#  The cat goes meow
#
# [[The, dog, is blue], [the, cat, goes, meow]]

sent_list = sent_tokenize(input_text)

data = []
for sent in sent_list:
    word_list = word_tokenize(sent)
    data.append(word_list)

model = Word2Vec(data, window = 5, sg = 1)

print(model.most_similar('skin'))