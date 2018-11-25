FROM python:3.6-slim-stretch

RUN apt-get -y update
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-numpy \
    zip \
    libboost-dev \
    libboost-all-dev \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

ADD requirements.txt /root/requirements.txt

RUN pip3.6 install torch

RUN export CPLUS_INCLUDE_PATH=/usr/local/include/python3.6m && \
    pip3.6 install --upgrade pip==9.0.3 && \
    pip3.6 install -r /root/requirements.txt

VOLUME ["/root"]

WORKDIR /root

ENTRYPOINT ["python", "main.py", "-f", "samples/emma_input.jpg"]
