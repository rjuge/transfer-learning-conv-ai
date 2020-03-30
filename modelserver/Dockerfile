FROM ubuntu:18.04

########################################  BASE SYSTEM
# set noninteractive installation
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y apt-utils
RUN apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    tzdata \
    curl

######################################## PYTHON3
RUN apt-get install -y \
    python3 \
    python3-pip


# transfer-learning-conv-ai
ENV PYTHONPATH /usr/local/lib/python3.6 
COPY . /app
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

