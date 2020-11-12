import requests
import pandas as pd
import json
import fasttext
import re
from bs4 import BeautifulSoup
import random



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

series = ["GSE27980",
        "GSE27980",
        "GSE12276",
        "GSE99394",
        "GSE66495",
        "GSE34970",
        "GSE111489",
        "GSE44660",
        "GSE52604",
        "GSE38057",
        "GSE66463",
        "GSE52050",
        "GSE14108",
        "GSE18544",
        "GSE76714",
        "GSE43837",
        "GSE73285",
        "GSE60464",
        "GSE57492",
        "GSE28049",
        "GSE17019",
        "GSE20016",
        "GSE26338",
        "GSE46928",
        "GSE69042",
        "GSE134026",
        "GSE125989",
        "GSE44354",
        "GSE23019",
        "GSE28313",
        "GSE74968",
        "GSE79534",
        "GSE117453",
        "GSE70576",
        "GSE98298",
        "GSE50493",
        "GSE114627",
        "GSE116531",
        "GSE38283",
        "GSE24100",
        "GSE23655",
        "GSE67088",
        "GSE51395",
        "GSE51411",
        "GSE43278",
        "GSE86501",
        "GSE103935"]

addToFile = open('Models/allQueries.txt', 'a+')
for ser in series:
    strToWrite = "{}\n".format(ser)
    addToFile.write(strToWrite)
addToFile.close()

nameFile = open("/Models/Queries/q5/names.txt", "w")
namesToQuery = []
abstracts = []

for s in series:
    api_text = ("http://stargeo.org/api/v2/series/{}/".format(s))
    rq = requests.get(api_text).text
    data = json.loads(rq)
    temp_dict = data['attrs']
    name = data['gse_name']
    summary = temp_dict['summary']
    if summary not in abstracts:
        abstracts.append(summary)
        namesToQuery.append(name)
    abstracts.append(summary)
    title = temp_dict['title']
    summary = summary.replace("\n", " ")
    title = title.replace("\n", " ")
    outFile = open("/Models/Queries/q5/{}.txt".format(name), "w")
    outFile.write(cleanText(title))
    outFile.write(' ')
    outFile.write(cleanText(summary))

ranNames = random.choices(namesToQuery, k = 3)
nameQueryFile = open("/Models/Queries/q5/names_to_query.txt", "w")

for name in ranNames:
    nameQueryFile.write(name + ' ')

for name in series:
    if name not in ranNames:
        nameFile.write(name + ' ')
