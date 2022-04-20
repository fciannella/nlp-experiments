# GPT-J on FT+Triton

## Introduction

First of all we need to prepare a triton image that contains the FT as a backend. After that we can work on the GPT-J model.

### Build the tritonserver container with the ft backend

We will do all the work into a working directory, initially empty, which we will call the WORKSPACE directory. In this case I will use the "triton-ft-gpt-j" directory.

```
mkdir /mnt/nvdl/usr/fciannella/src/triton-ft-gpt-j
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

The above will generate a container with triton and ft. You can push it to your registry if needed

#### Pushing to the gitlab docker registry

TBD

### Prepare the triton model store with GPT-J

We now need to create the model store with gpt-j. This means that we need to download gpt-j and convert its weigths so that they are compatible with FT, and place them in the proper directory that will be mounted inside the docker container we created above.

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

Now we need to run the command to convert the gpt-j weights. Not always one has access to the python and all the libraries, so we have two options, either we run the command in the host machine or inside a docker container. Therefore you can choose to follow one of the two next sections.

#### Convert the weights in the host machine

If you have all the libraries ready in your host VM then you can now simply run this command:

```
python3 ${WORKSPACE}/FasterTransformer/examples/pytorch/gptj/utils/gptj_ckpt_convert.py \
        --output-dir ${TRITON_MODELS_STORE}/fastertransformer/1 \
        --ckpt-dir ${WORKSPACE}/step_383500/ \
        --n-inference-gpus 4
```

#### Convert the weights inside the triton docker container

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

### Running the triton server on the 4X GPUs to serve the model on a single node

Normally you would think that one just needs to point the tritonserver to the model store, but in this case it is not as simple. There are a few steps. First of all let's start the triton container with the FT backend that we build previously

```
docker run -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864  --gpus=all -v ${WORKSPACE}:/ft_workspace ${TRITON_DOCKER_IMAGE} bash
```

This will land you into the triton container. 

In this container, go to the directory "/ft_workspace/fastertransformer_backend/all_models/gptj/". This is a directory that comes with the fastertransformer_backend repository and it is not the model_store we created and where we put the gpt-j weights. So it's a bit of a convoluted process, as we will point the triton server to this directory.

#### Adjust the config.pbtxt file

```
cd /ft_workspace/fastertransformer_backend/all_models/gptj/fastertransformer
vi config.pbtxt
```

You will have to change the `tensor_para_size` and `model_checkpoint_path` parameters

```
parameters {
  key: "tensor_para_size"
  value: {
    string_value: "4"
  }
}
```

and

```
parameters {
  key: "model_checkpoint_path"
  value: {
    string_value: "/ft_workspace/triton-model-store/fastertransformer/1/4-gpu"
  }
}
```

As you can see we are pointing the configuratin to the place where we had converted the weights, which we had called TRITON_MODELS_STORE before.

#### Run the tritonserver

We can now finally run the tritonserver and everything will come up

```
CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -n 1 --allow-run-as-root /opt/tritonserver/bin/tritonserver  --model-repository=/ft_workspace/fastertransformer_backend/all_models/gptj/ & 
```

In the above command the -n is the number of nodes and it should be 1 if we have one node, even if we have multiple GPUs on that node.

### Testing the model

This command will run an end to end test:

```
python3 /ft_workspace/fastertransformer_backend/tools/end_to_end_test.py
```

If you want to try your own prompt, you can edit the end_to_end_test.py file

