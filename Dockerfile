FROM python:3

COPY requirements.txt /
RUN pip install spacy
RUN python -m spacy download en_core_web_lg
RUN pip install -r requirements.txt

COPY NLTKImport.py /
COPY toIgnore/articles/* /
COPY getBioWordVecs.sh /
#COPY SpacyArticleAbstractAndTitle.py /
COPY runMe.sh /
COPY FastTextArticleTitle.py /
