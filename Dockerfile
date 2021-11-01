FROM nvcr.io/nvidia/pytorch:21.05-py3

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update
RUN apt-get install -y python3-dev build-essential lsb-release libffi-dev libxml2-dev libxslt-dev
RUN apt-get install -y python3-setuptools rustc cmake

ENV APP_HOME=/app
ENV PYTHONPATH=/app
ENV PORT=7001

RUN pip install -r requirements.txt

RUN mkdir -p /root/.jupyter
RUN cd /root/.jupyter && jupyter notebook --generate-config
RUN sed -i "s/# c.NotebookApp.token = '<generated>'/c.NotebookApp.token = \'\'/" /root/.jupyter/jupyter_notebook_config.py

COPY *.py /app/
COPY run_servers.sh /app
RUN chmod +x /app/run_servers.sh

ENTRYPOINT /app/run_servers.sh