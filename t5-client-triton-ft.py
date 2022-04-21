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
    #t5_model = T5ForConditionalGeneration.from_pretrained(args_dict["model"])
    tokenizer = T5Tokenizer.from_pretrained(args_dict["model"])
    fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(args_dict["model"])

    client_util = httpclient

    url = "nvdl-smc-02:8000"
    model_name = "fastertransformer"
    request_parallelism = 10
    verbose = False
    client = client_util.InferenceServerClient(
        url, concurrency=request_parallelism, verbose=verbose
    )
    t5_task_input = None
    while True:
        t5_task_input = input(
            "what do you want to do on T5?, format <task>: <sequence>, use exit to escape\n\n"
        )
        if t5_task_input == "exit":
            return
        else:
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

            print("sent request\n")
            result = client.infer(model_name, inputs)
            print("get request\n")

            ft_decoding_outputs = result.as_numpy("OUTPUT0")
            ft_decoding_seq_lens = result.as_numpy("OUTPUT1")
            # print(type(ft_decoding_outputs), type(ft_decoding_seq_lens))
            # print(ft_decoding_outputs, ft_decoding_seq_lens)
            tokens = fast_tokenizer.decode(
                ft_decoding_outputs[0][0][: ft_decoding_seq_lens[0][0]],
                skip_special_tokens=True,
            )
            print(tokens)
            print("\n")
            """print("output from T5 model using HF library:\n")
            input_ids = tokenizer(t5_task_input, return_tensors="pt").input_ids
            outputs = t5_model.generate(input_ids)
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            print("\n")"""


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
        default="t5-base",
        metavar="STRING",
        help="T5 model size.",
        choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"],
    )
    args = parser.parse_args()

    translate(vars(args))