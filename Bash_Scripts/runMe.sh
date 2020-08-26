#!/bin/bash

./getBioWordVecs.sh
python3 ./NLTKImport.py
#echo "ARTICLE TITLE ONLY"
#python3 ./FastTextArticleTitle.py
#echo "ARTICLE ABSTRACT ONLY"
#python3 ./FastTextArticleAbstract.py
#echo "ARTICLE TITLE AND ABSTRACT"
#python3 ./FastTextTitleAndAbstract.py
#./getConceptVecs.sh
#python3 ./bioConceptVec.py
#python3 ./mlNLP.py
#python3 ./getKeywords.py
#python3 ./testKeywords.py
#python3 getQuery1.py
#python3 getQuery2.py
#python3 getQuery3.py
#python3 getQuery4.py
#python3 getQuery5.py
python3 allKeywordBioWordVec.py
python3 getResults.py
#python3 geoAPI.py
