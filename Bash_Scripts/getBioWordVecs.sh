#!/bin/bash

#wget -O ./BioWordModel/model.bin https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin

FILE=/Models/model.bin
if [ -f "$FILE" ]; then
    echo "Model already downloaded!"
else
    wget -O ./Models/model.bin https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin
fi
