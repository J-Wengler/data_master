import requests
import pandas as pd
import json
import fasttext
import re
from bs4 import BeautifulSoup




def cleanText(text):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'http\S+', '', text)
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
series = ["GSE42639",
        "GSE42638",
        "GSE41088",
        "GSE43764",
        "GSE48890",
        "GSE36328",
        "GSE60572",
        "GSE60573"]
nameFile = open("/Models/Queries/q4/names.txt", "w")

for s in series:
    api_text = ("http://stargeo.org/api/v2/series/{}/".format(s))
    rq = requests.get(api_text).text
    data = json.loads(rq)
    temp_dict = data['attrs']
    name = data['gse_name']
    summary = temp_dict['summary']
    title = temp_dict['title']
    summary = summary.replace("\n", " ")
    title = title.replace("\n", " ")
    outFile = open("/Models/Queries/q4/{}.txt".format(name), "w")
    outFile.write(cleanText(title))
    outFile.write(' ')
    outFile.write(cleanText(summary))
    nameFile.write(name)
    nameFile.write(' ')
nameFile.close()