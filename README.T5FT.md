## Build the FT Backend for Triton

```
mkdir ft_workspace
cd ft_workspace
 
export WORKSPACE=$(pwd)
export SRC_MODELS_DIR=${WORKSPACE}/models
export TRITON_MODELS_STORE=${WORKSPACE}/triton-model-store
export CONTAINER_VERSION=21.08
export TRITON_DOCKER_IMAGE=fciannella/nlp-experiments/triton_with_ft:${CONTAINER_VERSION}
 
git clone https://github.com/triton-inference-server/fastertransformer_backend.git -b dev/v1.1_beta
git clone https://github.com/triton-inference-server/server.git
git clone -b dev/v5.0_beta https://github.com/NVIDIA/FasterTransformer # Used for convert the checkpoint and triton output
ln -s server/qa/common .
cd fastertransformer_backend/
docker build --rm --build-arg TRITON_VERSION=${CONTAINER_VERSION} -t ${TRITON_DOCKER_IMAGE} -f docker/Dockerfile .
```


### Dump the weights

Make sure you are in the workspace directory:

```
/mnt/nvdl/usr/fciannella/src/ft_workspace
```

Now you can run the commands inside the docker container:

```
docker run -it --rm \
 --privileged \
 --gpus all \
 -v /var/run/docker.sock:/var/run/docker.sock \
 -v ${PWD}:${PWD} \
 -w ${PWD} \
 --net host \
 --name working_image \
 ${TRITON_DOCKER_IMAGE} /bin/bash
```

Now we will work inside the container in the ft directory:

```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs
git lfs install

cd fastertransformer_backend
git clone https://huggingface.co/t5-base 
pip install -r tools/t5_utils/t5_requirement.txt

## You are inside the /mnt/nvdl/usr/fciannella/src/ft_workspace/fastertransformer_backend directory

python ../FasterTransformer/examples/pytorch/t5/utils/t5_ckpt_convert.py \
  -o all_models/t5/fastertransformer/1/ -i t5-base/ -infer_gpu_num 1
```

Now copy the weights one level up:

```
cd /mnt/nvdl/usr/fciannella/src/ft_workspace/fastertransformer_backend/all_models/t5/fastertransformer/1/1-gpu
mv * ../
cd ../
rm -rf 1-gpu
```

Now we need to edit the config.pbtxt file, make sure that at the end of the file you have a stanza looking like this:

```
cd /mnt/nvdl/usr/fciannella/src/ft_workspace/fastertransformer_backend/all_models/t5/fastertransformer

vi config.pbtxt 

# Now edit the file and then save

parameters {
  key: "model_checkpoint_path"
  value: {
    string_value: "/mnt/nvdl/usr/fciannella/src/ft_workspace/fastertransformer_backend/all_models/t5/fastertransformer/1"
  }
} 
```

You can now finally run the server (from inside the container, because you are inside it already):

```
tritonserver --model-repository=/mnt/nvdl/usr/fciannella/src/ft_workspace/fastertransformer_backend/all_models/t5/fastertransformer
```