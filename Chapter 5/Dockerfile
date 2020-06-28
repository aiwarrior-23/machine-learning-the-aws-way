
FROM python:3.7-slim-buster

RUN pip3 install pandas==0.25.3 scikit-learn==0.21.3
ENV PYTHONUNBUFFERED=TRUE
ENV PATH="/opt/ml/code:${PATH}"

COPY preprocessing.py /opt/ml/code/preprocessing.py

WORKDIR /opt/ml/code