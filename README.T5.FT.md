# T5 on FT+Triton

## Introduction

First of all we need to prepare a triton image that contains the FT as a backend. After that we can work on the T5 model.

### Build the tritonserver container with the ft backend

We will do all the work into a working directory, initially empty, which we will call the WORKSPACE directory. In this case I will use the "triton-ft-t5" directory.

```
mkdir /mnt/nvdl/usr/fciannella/src/triton-ft-t5
cd /mnt/nvdl/usr/fciannella/src/triton-ft-t5
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

#### Convert the weights in the host machine

If you have access to python on the host machine and you can install libraries, then you can convert the weights using the following:

```
cd ${WORKSPACE}
pip install -r ${WORKSPACE}/fastertransformer_backend/tools/t5_utils/t5_requirement.txt
```

We can now convert the weights:

```
python3 ${WORKSPACE}/FasterTransformer/examples/pytorch/t5/utils/huggingface_t5_ckpt_convert.py \
        -in_file ${WORKSPACE}/t5-3b/ \
        -saved_dir ${WORKSPACE}/fastertransformer_backend/all_models/t5/fastertransformer/1/ \
        -infer_gpu_num 4
```

#### Convert the weights inside the triton docker container

In this case we launch the docker container first and we convert the weights inside the container

```
cd ${WORKSPACE}
docker run -ti -d --gpus all -v `pwd`:`pwd` --name triton-convert-weights ${TRITON_DOCKER_IMAGE} 
docker exec -ti triton-convert-weights /bin/bash
```

This will land you inside the container. Now we need to reissue the environment variable again:

```
cd /mnt/nvdl/usr/fciannella/src/triton-ft-t5/
export WORKSPACE=$(pwd)
export SRC_MODELS_DIR=${WORKSPACE}/models
export TRITON_MODELS_STORE=${WORKSPACE}/triton-model-store
pip install -r ${WORKSPACE}/fastertransformer_backend/tools/t5_utils/t5_requirement.txt
```

Finally we can convert the weights:

```
python3 ${WORKSPACE}/FasterTransformer/examples/pytorch/t5/utils/huggingface_t5_ckpt_convert.py \
        -in_file ${WORKSPACE}/t5-3b/ \
        -saved_dir ${WORKSPACE}/fastertransformer_backend/all_models/t5/fastertransformer/1/ \
        -infer_gpu_num 4
```

Here is an example of the output:

```
root@1eb76eb7e553:/mnt/nvdl/usr/fciannella/src/triton-ft-t5# python3 ${WORKSPACE}/FasterTransformer/examples/pytorch/t5/utils/huggingface_t5_ckpt_convert.py         -in_file ${WORKSPACE}/t5-3b/         -saved_dir ${WORKSPACE}/fastertransformer_backend/all_models/t5/fastertransformer/1/         -infer_gpu_num 4

=============== Argument ===============
saved_dir: /mnt/nvdl/usr/fciannella/src/triton-ft-t5/fastertransformer_backend/all_models/t5/fastertransformer/1/
in_file: /mnt/nvdl/usr/fciannella/src/triton-ft-t5/t5-3b/
infer_gpu_num: 4
weight_data_type: fp32
========================================
[INFO] Spend 0:04:04.728308 (h:m:s) to convert the model
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

#### Adjust the config.pbtxt file

```
cd /ft_workspace/fastertransformer_backend/all_models/t5/fastertransformer
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
    string_value: "/ft_workspace/fastertransformer_backend//all_models/t5/fastertransformer/1/4-gpu"
  }
}
```

As you can see we are pointing the configuratin to the place where we had converted the weights, which we had called TRITON_MODELS_STORE before.

#### Run the tritonserver

We can now finally run the tritonserver and everything will come up

```
CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -n 1 --allow-run-as-root /opt/tritonserver/bin/tritonserver  --model-repository=/ft_workspace/fastertransformer_backend/all_models/t5 
```

In the above command the -n is the number of nodes and it should be 1 if we have one node, even if we have multiple GPUs on that node.

### Testing the model

We need first to add some client libraries:

```
pip install transformers
pip install sacrebleu
pip install sentencepiece
```

Finally we can run this command that will execute an end to end translation test:

```
python3 /ft_workspace/fastertransformer_backend/tools/t5_utils/t5_end_to_end_test.py --batch_size 32 --source /ft_workspace/fastertransformer_backend/tools/t5_utils/test.en --target /ft_workspace/fastertransformer_backend/tools/t5_utils/test.de
```