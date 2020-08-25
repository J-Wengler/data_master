FROM python:3

COPY requirements.txt /
RUN pip install spacy
RUN pip install git+https://github.com/boudinfl/pke.git
RUN python -m spacy download en
RUN python -m spacy download en_core_web_lg
RUN pip install -r requirements.txt
COPY Python_Code/* /
#COPY toIgnore/articles/* /
COPY Bash_Scripts/* /
