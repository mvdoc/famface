bootstrap: docker
from: neurodebian:jessie

%post
    apt-get update
    apt-get install -y eatmydata wget
    wget -O- http://neuro.debian.net/lists/jessie.us-nh.full | tee /etc/apt/sources.list.d/neurodebian.sources.list
    apt-key adv --recv-keys --keyserver hkp://pool.sks-keyservers.net:80 0xA5D32F012649A5A9
    apt-get update
    eatmydata apt-get install -y \
      python-mvpa2 fsl-core ants python-pip python-datalad python-scipy python-numpy \
      python-nipype=0.11.0-1~nd80+1 python-sklearn python-dateutil
    pip install joblib

    mkdir /data /scripts /derivatives /ihome /idata

%runscript
    echo "### Running container for famface ###"
    exec /bin/bash
