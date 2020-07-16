import requests
import pandas as pd
import fasttext
import re
from bs4 import BeautifulSoup
import os
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
import io
import wikipedia
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt
from TextRank import TextRank4Keyword

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
    #tokens = [stemmer.lemmatize(word) for word in tokens]
    #tokens = [word for word in tokens if word not in en_stop]
    #tokens = [word for word in tokens if len(word) > 3]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def getDocEmbedding(text, model):

    #text = cleanText(text)

    tr4w = TextRank4Keyword()
    tr4w.analyze(text, candidate_pos = ['NOUN', 'PROPN','ADJ'], window_size=6, lower=False)
    ans = tr4w.get_keywords(10)
    if ans is not None:
        allWords = []
        for key in ans:
            allWords.append(key)
        first_word = allWords[0]
        doc_vec = model[first_word]
        numWords = 1
        for i, word in enumerate(allWords):
            if (i != 0):
                new_vec = model[word]
                if new_vec is not None:
                    numWords += 1
                    doc_vec = np.add(doc_vec, new_vec)
        doc_vec = doc_vec / numWords
        return doc_vec
    else:
        allWords = word_tokenize(text)
        first_word = allWords[0]
        doc_vec = model[first_word]
        numWords = 1
        for i, word in enumerate(allWords):
            if (i != 0):
                new_vec = model[word]
                if new_vec is not None:
                    numWords += 1
                    doc_vec = np.add(doc_vec, new_vec)
        doc_vec = doc_vec / numWords
        return doc_vec
# Fetch first 10 series, defaults to 100
#r = requests.get('http://stargeo.org/api/v2/series/?limit=10')
#assert r.ok
#data = r.json()

series_to_summary = {}

data = pd.read_json('http://stargeo.org/api/v2/series/?limit=100000')
for row in data['results']:
    temp_dict = row['attrs']
    name = row['gse_name']
    summary = temp_dict['summary']
    title = temp_dict['title']
    series_to_summary[name] = [summary,title]

test_text = "	Phase I study of Veliparib in combination with Cisplatin and Vinorelbine for patients with advanced triple-negative breast cancer (TNBC) and/or BRCA mutation-associated breast cancer 	PURPOSE: Preclinically, cisplatin is synergistic with vinorelbine and the poly(ADP-ribose) polymerase (PARP) inhibitor veliparib, and has anti-neoplastic activity in TNBC and BRCA mutation-associated breast cancer. This phase I study assessed veliparib with cisplatin and vinorelbine. PATIENTS AND METHODS: A 3+3 dose escalation design evaluated veliparib administered BID for 14 days with cisplatin (75 mg/m2 day 1) and vinorelbine (25 mg/m2 days 1,8) every 21 days, for six to ten cycles, followed by veliparib monotherapy. Pharmacokinetics, measurement of poly(ADP-ribose) in peripheral blood mononuclear cells, and preliminary efficacy were assessed. Immunohistochemistry and gene expression profiling were performed to evaluate potential predictors of response. 	Efficacy of carboplatin alone and in combination with ABT888 in intracranial murine models of BRCA-mutated and BRCA-wild-type triple negative breast cancer Purpose:Triple negative breast cancer (TNBC) commonly metastasizes to the brain and predicts poor prognosis with limited therapeutic options. TNBC frequently harbors BRCA mutations translating to platinum sensitivity; platinum response may be augmented by additional suppression of DNA repair mechanisms through poly(ADP-ribose)polymerase (PARP) inhibition. We evaluated brain penetrance and efficacy of Carboplatin +/- the PARP inhibitor ABT888, and investigated gene expression changes in murine intracranial (IC) TNBC models stratified by BRCA and molecular subtype status. Experimental design:Athymic mice were inoculated intra-cerebrally with BRCA-mutant: SUM149 (basal), MDA-MB-436 (claudin-low), or BRCA-wild-type: MDA-MB-468 (basal), MDA-MB-231BR (claudin-low) TNBC cells and treated with PBS control (IP, weekly), Carboplatin (50mg/kg/week, IP), ABT888 (25mg/kg/day, OG), or their combination. DNA-damage (?-H2AX) and apoptosis (cleaved-Caspase-3(cC3)) were assessed via IHC of IC tumors. Gene expression of BRCA-mutant IC tumors was measured. Results: Carboplatin+/-ABT888 significantly improved survival in BRCA-mutant IC models compared to control, but did not improve survival in BRCA-wild-type IC models. Carboplatin+ABT888 revealed a modest survival advantage versus Carboplatin in BRCA-mutant models. ABT888 yielded a marginal survival benefit in the MDA-MB-436 but not in the SUM149 model. BRCA-mutant SUM149 expression of ?-H2AX and cC3 proteins was elevated in all treatment groups compared to Control, while BRCA-wild-type MDA-MB-468 cC3 expression did not increase with treatment. Carboplatin treatment induced common gene expression changes in BRCA-mutant models.Conclusions: Carboplatin+/-ABT888 improves survival in BRCA-mutant IC TNBC models with corresponding DNA damage and gene expression changes. Combination therapy represents a promising treatment strategy for patients with TNBC brain metastases warranting further clinical investigation. 	mTOR inhibitors suppress homologous recombination repair and synergize with PARP inhibitors via regulating SUV39H1 in BRCA-proficient triple-negative breast cancer 	This analysis was used to search for genes that were differentially expressed between cells treated with DMSO and those treated with the mTOR inhibitor everolimus."

model = fasttext.load_model('/BioWordModel/model.bin')
initial_doc = getDocEmbedding(test_text, model)

percent_to_name = {}

for key in series_to_summary:
    title_summary = series_to_summary[key]
    full_str = cleanText(title_summary[0]) + cleanText(title_summary[1])
    if(len(full_str) > 0):
        doc2 = getDocEmbedding(full_str, model)
        cosine_similarity = 1 - cosine(initial_doc, doc2)
        cosine_similarity *= 100
        percent_to_name[cosine_similarity] = title_summary[1]

print()

matchingArticles = ["Phase I study of Veliparib in combination with Cisplatin and Vinorelbine for patients with advanced triple-negative breast cancer (TNBC) and/or BRCA mutation-associated breast cancer","Efficacy of carboplatin alone and in combination with ABT888 in intracranial murine models of BRCA-mutated and BRCA-wild-type triple negative breast cancer", "mTOR inhibitors suppress homologous recombination repair and synergize with PARP inhibitors via regulating SUV39H1 in BRCA-proficient triple-negative breast cancer", "TBCRC 018:  Phase II study of iniparib in combination with irinotecan to treat progressive triple negative breast cancer brain metastases", "SUM159 BrCa Decitabine Gene Expression", "Microarray analysis of differentially expressed genes in ovarian and fallopian tube epithelium from risk-reducing salpingo-oophorectomies"]

avg_percent = 0
num_percent = 0
print()
for percent in percent_to_name:
    avg_percent += percent
    num_percent += 1
    if percent_to_name[percent] in matchingArticles:
        print("{} : {}%".format(percent_to_name[percent], percent))
print()
print("AVERAGE SIMILARITY : {}%".format(avg_percent / num_percent))

