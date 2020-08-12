docker build -t data_master .

#docker run -i -t --rm data_master /helloworld.py

mkdir -p /Models

#docker --user $(id -u):$(id -g)
docker volume create data-volume
#docker --user $(id -u):$(id -g) 
docker run --rm -i -t -v ~/WORDEMBEDDING/data_master/Models/:/Models/ data_master /runMe.sh

#docker run --rm -i -t -v "$(pwd)":/BioWordModel data_master /runMe.sh

#docker run --rm -i -t data_master ./runMe.sh
