from nltk.tokenize import sent_tokenize
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

sent_list = sent_tokenize(input_text)

with open("/Users/jameswengler/PycharmProjects/WordEmbedding/processedFile.txt", "w+") as out:
    for sent in sent_list:
        out.write(sent)
    out.close()