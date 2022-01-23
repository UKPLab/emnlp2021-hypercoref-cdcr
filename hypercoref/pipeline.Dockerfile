# syntax=docker/dockerfile:1

FROM python:3.7.12-bullseye

RUN apt update && \
    apt install -y nano vim less screen tmux unzip wget locales p7zip-full && \
    locale-gen en_US.UTF-8
COPY . /hypercoref
WORKDIR /hypercoref

# set locale to make python Click happy
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# python dependencies - we need to install hlwy-lsh *after* numpy, otherwise it can't find the numpy header files...
RUN pip install --upgrade pip==21.3.1 wheel && \
    pip install $(grep -v "^ *#\|^hlwy-lsh" /hypercoref/resources/requirements/baked_2021-11-22.txt) && \
    pip install $(grep "^hlwy-lsh" /hypercoref/resources/requirements/baked_2021-11-22.txt)

ENTRYPOINT ["/bin/bash"]