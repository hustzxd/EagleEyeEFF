FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
MAINTAINER zxd
RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y screen
RUN apt-get install -y htop
RUN apt-get install -y zsh
RUN apt-get install -y openssh-server
RUN apt-get install -y curl
RUN apt-get install -y git
RUN apt-get install -y protobuf-compiler
RUN apt-get install -y iputils-ping
WORKDIR /workspace
# COPY requirements.txt requirements.txt
# RUN pip install -r requirements.txt