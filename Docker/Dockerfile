FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

WORKDIR /csm-streaming-api

RUN apt-get update
RUN apt-get install python3.12 -y
RUN apt-get install python-is-python3 -y
RUN apt-get install python3-pip -y
RUN apt-get install python3-venv -y
RUN apt-get install git -y
RUN apt-get install curl -y
#RUN apt-get install ffmpeg -y
RUN apt-get install libportaudio2  -y
RUN apt-get install portaudio19-dev -y

RUN git clone https://github.com/nobandegani/csm-streaming-api.git --single-branch .

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && pip install -r requirements.txt

ARG HUGGINGFACE_TOKEN
RUN huggingface-cli login --token $HUGGINGFACE_TOKEN

RUN python setup.py