FROM python:3

COPY /requirements.txt .
#COPY . . 
COPY /FastTextArticleTitle.py .
COPY /helloworld.py .
COPY /runMe.sh .

RUN pip install spacy
RUN python -m spacy download en_core_web_lg
RUN pip install -r requirements.txt
