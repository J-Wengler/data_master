import fasttext
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
model = fasttext.load_model("/BioWordModel/model.bin")

def getWordVec(word):
    return model[word]

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
    tokens = [stemmer.lemmatize(word, pos = 'v') for word in tokens]
    tokens = [word for word in tokens if word not in en_stop]
    tokens = [word for word in tokens if len(word) > 3]

    preprocessed_text = ' '.join(tokens)
    print(preprocessed_text)
    return preprocessed_text



word_list = []
vec_list = []
for i in range(1,11):
    inString = '/Article{}Name.txt'.format(i)
    in_file = open(inString)
    in_text = in_file.read()
    in_text = cleanText(in_text)
    allWords = word_tokenize(in_text)
    for word in  allWords:
        word_list.append(word)
        vec_list.append(getWordVec(word))

word_list = pd.Series(word_list)
vec_list = pd.Series(vec_list)

pd.set_option('display.max_rows', 150)

df = pd.concat([word_list, vec_list], axis = 1)
df.columns = ['Word', 'Embedding']

print(df)


