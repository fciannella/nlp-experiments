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
docker run -d -ti --gpus device=1 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --network=host --ipc=host -it -v /mnt/nvdl/usr/fciannella/src/ft_workspace/:/workspace/t5_triton --name=triton_server ${TRITON_DOCKER_IMAGE}

docker exec -ti triton_server /bin/bash
```


Now we will work inside the container in the ft directory:

```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs
git lfs install

cd t5_triton
git clone https://huggingface.co/t5-base 

cd fastertransformer_backend
pip install -r tools/t5_utils/t5_requirement.txt

# The command below has to be run inside the same directory in which you saved the t5-base files

cd .. # you are now in /workspace/t5_triton
python ./FasterTransformer/examples/pytorch/t5/utils/t5_ckpt_convert.py  -o fastertransformer_backend/all_models/t5/fastertransformer/1/ -i t5-base/ -infer_gpu_num 1

```

Now copy the weights one level up:

```
cd /workspace/t5_triton/fastertransformer_backend/all_models/t5/fastertransformer/1/1-gpu
mv * ../
cd ../
rm -rf 1-gpu
```

Now we need to edit the config.pbtxt file, make sure that at the end of the file you have a stanza looking like this:

```
cd /workspace/t5_triton/fastertransformer_backend/all_models/t5/fastertransformer

vi config.pbtxt 

# Now edit the file and then save

parameters {
  key: "model_checkpoint_path"
  value: {
    string_value: "/workspace/t5_triton/fastertransformer_backend/all_models/t5/fastertransformer/1"
  }
} 
```

You can now finally run the server (from inside the container, because you are inside it already):

```
cd /workspace/t5_triton/

tritonserver --model-repository=/workspace/t5_triton/fastertransformer_backend/all_models/t5
```

## Testing the model

You need to use this script from inside the container eventually:

```python
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
from datetime import datetime
import numpy as np
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../")

from transformers import PreTrainedTokenizerFast
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
)  # transformers-4.10.0-py3
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


def translate(args_dict):
    torch.set_printoptions(precision=6)
    batch_size = args_dict["batch_size"]

    t5_model = T5ForConditionalGeneration.from_pretrained(args_dict["model"])

    tokenizer = T5Tokenizer.from_pretrained(args_dict["model"])
    fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(args_dict["model"])

    client_util = httpclient

    url = "localhost:8000"
    model_name = "fastertransformer"
    request_parallelism = 10
    verbose = False
    with client_util.InferenceServerClient(
        url, concurrency=request_parallelism, verbose=verbose
    ) as client:
        t5_task_input = input(
            "what do you want to do on T5?, format <task>: <sequence>\n"
        )
        # t5_task_input="translate English to German: a man was found dead in a cave"
        print(t5_task_input)
        sys.stdout.flush()
        input_token = tokenizer(t5_task_input, return_tensors="pt", padding=True)
        input_ids = input_token.input_ids.numpy().astype(np.uint32)
        mem_seq_len = (
            torch.sum(input_token.attention_mask, dim=1).numpy().astype(np.uint32)
        )
        mem_seq_len = mem_seq_len.reshape([mem_seq_len.shape[0], 1])
        inputs = [
            client_util.InferInput(
                "INPUT_ID", input_ids.shape, np_to_triton_dtype(input_ids.dtype)
            ),
            client_util.InferInput(
                "REQUEST_INPUT_LEN",
                mem_seq_len.shape,
                np_to_triton_dtype(mem_seq_len.dtype),
            ),
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(mem_seq_len)

        print("set request")
        result = client.infer(model_name, inputs)
        print("get request")

        ft_decoding_outputs = result.as_numpy("OUTPUT0")
        ft_decoding_seq_lens = result.as_numpy("OUTPUT1")
        print(type(ft_decoding_outputs), type(ft_decoding_seq_lens))
        print(ft_decoding_outputs, ft_decoding_seq_lens)
        tokens = fast_tokenizer.decode(
            ft_decoding_outputs[0][0][: ft_decoding_seq_lens[0][0]],
            skip_special_tokens=True,
        )
        print(tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-batch",
        "--batch_size",
        type=int,
        default=1,
        metavar="NUMBER",
        help="batch size (default: 1)",
    )
    parser.add_argument(
        "-model",
        "--model",
        type=str,
        default="t5-small",
        metavar="STRING",
        help="T5 model size.",
        choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"],
    )
    args = parser.parse_args()

    translate(vars(args))

```

Then you can run from inside the /workspace/t5_triton directory in the container:

```angular2html
root@nvdl-smc-03:/workspace/t5_triton# python client.py --model t5-base
The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization.
The tokenizer class you load from this checkpoint is 'T5Tokenizer'.
The class this function is called from is 'PreTrainedTokenizerFast'.
what do you want to do on T5?, format <task>: <sequence>
```

For instance you can try a translation:

```angular2html
Translate EnglishTo German: I want to rest a bit!
set request
get request
<class 'numpy.ndarray'> <class 'numpy.ndarray'>
[[[ 1674  6509   236  4499 17177    55     1  1410  3523     1     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0
       0     0     0     0     0     0     0     0     0     0     0]]] [[7]]
Ich möchte ein wenig Ruhe!
root@nvdl-smc-03:/workspace/t5_triton#
```