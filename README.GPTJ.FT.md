# GPT-J on FT+Triton

## Introduction

First of all we need to prepare a triton image that contains the FT as a backend. After that we can work on the GPT-J model.

### Prepare the triton container

```
cd /mnt/nvdl/usr/fciannella/src/triton-ft-gpt-j
export WORKSPACE=$(pwd)
export SRC_MODELS_DIR=${WORKSPACE}/models
export TRITON_MODELS_STORE=${WORKSPACE}/triton-model-store
export CONTAINER_VERSION=22.03
export TRITON_DOCKER_IMAGE=triton_with_ft:${CONTAINER_VERSION}
## Build the container
```

We can build the triton server with the fastertransformer backend now

```
cd ${WORKSPACE}
git clone https://github.com/triton-inference-server/fastertransformer_backend
git clone https://github.com/triton-inference-server/server.git 
git clone https://github.com/NVIDIA/FasterTransformer.git
ln -s server/qa/common .
cd fastertransformer_backend
docker build --rm   \
    --build-arg TRITON_VERSION=${CONTAINER_VERSION}   \
    -t ${TRITON_DOCKER_IMAGE} \
    -f docker/Dockerfile \
    .
```

The above will generate a container with triton and ft.

### Prepare the triton model store with GPT-J

```
export WORKSPACE=$(pwd)
export SRC_MODELS_DIR=${WORKSPACE}/models
export TRITON_MODELS_STORE=${WORKSPACE}/triton-model-store
cd ${WORKSPACE}
git clone https://github.com/NVIDIA/FasterTransformer.git # Used for convert the checkpoint and triton output
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -P models
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -P models
wget https://mystic.the-eye.eu/public/AI/GPT-J-6B/step_383500_slim.tar.zstd
tar -I zstd -xf step_383500_slim.tar.zstd -C ${SRC_MODELS_DIR}

mkdir ${TRITON_MODELS_STORE}/fastertransformer/1 -p
```

Now we need to run the command to convert the gpt-j weights. Not always one has access to the python and all the libraries, so we will run this inside the triton container

```
docker run -ti -d --gpus all -v `pwd`:`pwd` --name triton-convert-weights ${TRITON_DOCKER_IMAGE} 
docker exec -ti triton-convert-weights /bin/bash
```

This will land you inside the container. Now we need to reissue the environment variable again:

```
cd /mnt/nvdl/usr/fciannella/src/triton-ft-gpt-j
export WORKSPACE=$(pwd)
export SRC_MODELS_DIR=${WORKSPACE}/models
export TRITON_MODELS_STORE=${WORKSPACE}/triton-model-store

pip install jax
pip install jaxlib
```

Finally we can convert the weights:

```
python3 ${WORKSPACE}/FasterTransformer/examples/pytorch/gptj/utils/gptj_ckpt_convert.py \
        --output-dir ${TRITON_MODELS_STORE}/fastertransformer/1 \
        --ckpt-dir ${WORKSPACE}/step_383500/ \
        --n-inference-gpus 4
```

Once the weights have been translated you can now exit the container

```
exit
```

### Running the triton server on the 4X GPUs to serve the model










# This is now obsolete DO NOT READ FURTHER

docker build -f Dockerfile.GPTJ -t fciannella/gptj:1.0 -t fciannella/gptj:latest .

docker run -d -ti --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --network=host --ipc=host -it -v /mnt/nvdl/usr/fciannella/src/gptj-data/:/gptj-data --name=gptj-container fciannella/gptj:latest

```


## Run GPT-J in Triton

### Prepare the data
```
docker exec -ti gptj-container /bin/bash
```

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