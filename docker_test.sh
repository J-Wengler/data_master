docker build -t data_master .

#docker run -i -t --rm data_master /helloworld.py

mkdir -p BioWordModel

./getBioWordVecs.sh 

docker volume create data-volume
docker run --rm -i -t -v ~/WORDEMBEDDING/data_master/BioWordModel/:/BioWordModel/ data_master /runMe.sh

#docker run --rm -i -t -v "$(pwd)":/BioWordModel data_master /runMe.sh

#docker run --rm -i -t data_master ./runMe.sh
