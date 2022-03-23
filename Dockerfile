FROM nvcr.io/nvidia/pytorch:22.02-py3

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update
RUN apt-get install -y python3-dev build-essential lsb-release libffi-dev libxml2-dev libxslt-dev
RUN apt-get install -y python3-setuptools rustc cmake
RUN apt-get install -y libsndfile1 ffmpeg

ENV APP_HOME=/app
ENV PYTHONPATH=/app
ENV PORT=7001

# COPY *.py /app/
COPY *.txt /app
COPY run_servers.sh /app
RUN chmod +x /app/run_servers.sh

RUN pip install -r requirements.txt
RUN pip install nemo_toolkit['all']

RUN mkdir -p /root/.jupyter
RUN cd /root/.jupyter && jupyter notebook --generate-config
RUN sed -i "s/# c.NotebookApp.token = '<generated>'/c.NotebookApp.token = \'\'/" /root/.jupyter/jupyter_notebook_config.py

ENTRYPOINT /app/run_servers.sh
