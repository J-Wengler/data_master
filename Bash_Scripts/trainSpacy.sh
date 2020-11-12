#!/bin/bash
#python -m spacy init-model en /Models/SPACY_SKIPGRAM --vectors-loc /Models/FT_STARGEO_SKIPGRAM.bin.trainables.vectors_ngrams_lockf.npy
python -m spacy pretrain /Models/starGEO.txt en_vectors_web_lg /Models/bert-model/
