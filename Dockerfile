FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /app

COPY . /app

RUN pip install jupyterlab
RUN pip install tqdm
RUN pip install torchtext
RUN pip install spacy
RUN spacy download pl_core_news_sm