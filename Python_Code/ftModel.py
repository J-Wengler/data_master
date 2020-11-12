from gensim.models.fasttext import FastText as FT_gensim
from gensim.test.utils import datapath
import requests
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

def generateText(test_set):
    all_text = ""
    #rq = requests.get('http://stargeo.org/api/v2/series/?limit=5').json()
    rq = requests.get('http://stargeo.org/api/v2/series/?limit=10000000').json()
    #data = pd.read_json(rq)
    series_to_summary = {}
    num = 0
    for row in rq['results']:
        temp_dict = row['attrs']
        name = row['gse_name'].rstrip()
        summary = temp_dict['summary']
        title = temp_dict['title']
        full_text = summary + title
        all_text += full_text

    return all_text

test_set = []
test_file = open('/Models/allQueries.txt')
for line in test_file:
    test_set.append(line.rstrip())

all_text = generateText(test_set)

outFile = open('Models/starGEO.txt', 'w+')

sents = sent_tokenize(all_text)
tokenized_text = []
for sent in sents:
    temp_words = word_tokenize(sent)
    for word in temp_words:
        outFile.write(word + ' ')
