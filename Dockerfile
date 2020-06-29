FROM python:3

WORKDIR /Users/jameswengler/PycharmProjects/WordEmbedding

COPY . .

RUN pip install spacy
RUN python -m spacy download en_core_web_lg
RUN pip install -r requirements.txt


#RUN python3 FastTextTitleAndAbstract.py
RUN python3 SpacyArticleTitle.py
