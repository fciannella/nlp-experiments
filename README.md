# nlp-experiments

## Docker commands to build and run

```
docker build -t nlp-experiments:1.0.0 -t nlp-experiments:latest . 

docker run --gpus device=0 -d -it --rm --name nlp-experiments -h nlp-experiments -v /tmp:/tmp -v /mnt/nvdl/usr/fciannella/NeMo:/app/src/NeMo -v /mnt/nvdl/usr/fciannella/src/nlp-experiments:/app/src --shm-size=8g \
-p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
stack=67108864 --device=/dev/snd nlp-experiments:latest

docker run -d -it --rm --name nlp-experiments -h nlp-experiments -v /tmp:/tmp -v /Users/fciannella/PycharmProjects/nlp-experiments:/app/src --shm-size=8g \
-p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
stack=67108864 nlp-experiments:latest
```


## Experimenting with Triton


### Model Analyzer
```
docker pull nvcr.io/nvidia/tritonserver:21.10-py3-sdk

docker run -it --gpus all \
      -v /var/run/docker.sock:/var/run/docker.sock \
      -v /mnt/nvdl/usr/fciannella/models:/models \
      --net=host nvcr.io/nvidia/tritonserver:21.10-py3-sdk
```

## Model Navigator
```
docker build -t model-navigator .

docker run -it --rm \
 --gpus device=1 \
 -v /var/run/docker.sock:/var/run/docker.sock \
 -v /tmp:/tmp \
 -v ${PWD}:${PWD} \
 -w ${PWD} \
 --net host \
 --name model-navigator \
 model-navigator /bin/bash
```