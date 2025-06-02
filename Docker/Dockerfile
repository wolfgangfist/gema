FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

WORKDIR /csm-streaming-api

RUN apt-get update
RUN apt-get install python3.10 -y
RUN apt-get install python-is-python3 -y
RUN apt-get install pip -y
RUN apt-get install git -y
RUN apt-get install curl -y
RUN apt-get install ffmpeg -y
RUN apt-get libportaudio2  -y
RUN apt-get libportaudio-dev -y

RUN git clone https://github.com/nobandegani/csm-streaming-api.git --single-branch .
RUN pip install -r requirements.txt

ARG HUGGINGFACE_TOKEN
RUN huggingface-cli login --token $HUGGINGFACE_TOKEN

# setup
RUN python setup.py

# start
CMD python main.py