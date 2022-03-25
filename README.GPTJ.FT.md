## Build the container

```
docker build -f Dockerfile.GPTJ -t fciannella/gptj:1.0 -t fciannella/gptj:latest .

docker run -d -ti --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --network=host --ipc=host -it -v /mnt/nvdl/usr/fciannella/src/gptj-data/:/gptj-data --name=gptj-container fciannella/gptj:latest

```


## Run GPT-J in Triton

### Prepare the data
Create a directory called gptj-data and cd into it and then start cloning the needed repositories

```
mkdir gptj-data
cd gptj-data
git clone https://gitlab-master.nvidia.com/bhsueh/fastertransformer_backend -b dev-gptj
git clone https://gitlab-master.nvidia.com/zehuanw/FasterTransformer        -b v5.0-dev-gptj
git clone https://github.com/triton-inference-server/server.git # We need some tools when we test this backend
ln -s server/qa/common .

```

### Build the container

```
docker build -f Dockerfile.gptjtriton -t fciannella/gptjtriton:1.0 -t fciannella/gptjtriton:latest .

docker run -d -ti --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --network=host --ipc=host -it -v /mnt/nvdl/usr/fciannella/src/gptj-data/:/gptj-data --name=gptj-container fciannella/gptjtriton:latest

```

Set up the environment variables:

```
export WORKSPACE=$(pwd)
export SRC_MODELS_DIR=${WORKSPACE}/models
export TRITON_MODELS_STORE=${WORKSPACE}/triton-model-store
export CONTAINER_VERSION=21.07
export TRITON_DOCKER_IMAGE=triton_with_ft:${CONTAINER_VERSION}
```


## Build the triton container

```
docker build -f Dockerfile.gptjtriton -t fciannella/gptjtriton:1.0 -t fciannella/gptjtriton:latest .

docker run -d -ti --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --network=host --ipc=host -it -v /mnt/nvdl/usr/fciannella/src/gptj-data/:/gptj-data --name=gptj-container fciannella/gptjtriton:latest

```