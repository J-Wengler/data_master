#!/bin/bash
FILE=/BioWordModel/concept_model.bin
if [ -f "$FILE" ]; then
    echo "Model already downloaded!"
else
    wget -O ./BioWordModel/concept_model.bin https://ftp.ncbi.nlm.nih.gov/pub/lu/BioConceptVec/bioconceptvec_fasttext.bin
fi    
