import requests
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
import multiprocessing

def cleanText(text):
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

    return text


def generateText():
    all_text = ""
    #rq = requests.get('http://stargeo.org/api/v2/series/?limit=5').json()
    rq = requests.get('http://stargeo.org/api/v2/series/?limit=1000000').json()
    #data = pd.read_json(rq)
    series_to_summary = {}
    for row in rq['results']:
        temp_dict = row['attrs']
        name = row['gse_name']
        summary = temp_dict['summary']
        title = temp_dict['title']
        full_text = summary + title
        all_text += full_text
        #full_text = cleanText(full_text)
        #series_to_summary[name] = full_text

    return all_text

all_text = generateText()
sents = sent_tokenize(all_text)
tokenized_text = []
for sent in sents:
    temp_words = word_tokenize(sent)
    tokenized_text.append(temp_words)


w2v = Word2Vec(tokenized_text, size = 300, window = 5, min_count = 5, negative = 15, iter = 10, workers = multiprocessing.cpu_count())

model = w2v.wv
result = model.similar_by_word('BRCA1')
print(result[:10])


print("Got All Articles!")
