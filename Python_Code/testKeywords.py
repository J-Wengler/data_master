from TextRank import TextRank4Keyword
from nltk.stem import WordNetLemmatizer
import nltk
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize

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

pathToDoc='/TitleSummary1.txt'
in_file = open(pathToDoc)
in_text = in_file.read()
in_text = cleanText(in_text)

tr4w = TextRank4Keyword()
tr4w.analyze(in_text, candidate_pos = ['NOUN', 'PROPN', 'ADJ',"VERB"], window_size=6, lower=False)
ans = tr4w.get_keywords(10)

for key in ans:
    if ans[key] > 1.0:
        print(key)
