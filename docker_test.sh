docker build -t data_master .

mkdir -p /Models

docker volume create data-volume

docker run --rm -i -t -v ~/WORDEMBEDDING/data_master/Models/:/Models/ data_master /runMe.sh
