docker build -t data_master .

#docker run -i -t --rm data_master /helloworld.py

mkdir -p BioWordModel
docker run --rm -i -t -v "$(pwd)":/BioWordModel data_master /runMe.sh

#docker run --rm -i -t data_master ./runMe.sh
