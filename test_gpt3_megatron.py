import json
import os
from argparse import ArgumentParser

import torch
from pytorch_lightning.trainer.trainer import Trainer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from nemo.collections.nlp.data.language_modeling.megatron.request_dataset import GPTRequestDataset
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import inject_model_parallel_rank

assert torch.cuda.is_available()

checkpoint_dir = "/5b_checkpoints/checkpoints"
checkpoint_name = "megatron_gpt--val_loss=1.78-step=32121-consumed_samples=46254240.0-last.ckpt"
# I have a node with 8 V100 GPUs so not sure what should I set for these variables
devices = 2
num_nodes = 1
# The below comes from the checkpoint
tensor_model_parallel_size=2
pipeline_model_parallel_size=1
precision=16
hparams_file=None

trainer = Trainer(
    plugins=[NLPDDPPlugin()],
    devices=devices,
    num_nodes=num_nodes,
    accelerator='gpu',
    precision=precision,
)

app_state = AppState()
if tensor_model_parallel_size > 1 or pipeline_model_parallel_size > 1:
    app_state.pipeline_model_parallel_size = pipeline_model_parallel_size
    app_state.tensor_model_parallel_size = tensor_model_parallel_size
    app_state.model_parallel_size = tensor_model_parallel_size * pipeline_model_parallel_size
    (
        app_state.tensor_model_parallel_rank,
        app_state.pipeline_model_parallel_rank,
        app_state.model_parallel_size,
        _,
    ) = fake_initialize_model_parallel(
        world_size=app_state.model_parallel_size,
        rank=trainer.global_rank,
        tensor_model_parallel_size_=app_state.tensor_model_parallel_size,
        pipeline_model_parallel_size_=app_state.pipeline_model_parallel_size,
    )
    # inject model parallel rank
checkpoint_path = inject_model_parallel_rank(os.path.join(checkpoint_dir, checkpoint_name))


model = MegatronGPTModel.load_from_checkpoint(checkpoint_path, hparams_file=hparams_file, trainer=trainer)

model.freeze()

def pad_collate(batch):
    tokens, tokens_to_generate = batch[0]['data'], batch[0]['tokens_to_generate']
    compute_logprobs = batch[0]['compute_logprobs']
    lens = [len(token) for token in tokens]

    tokens_pad = pad_sequence(tokens, batch_first=False, padding_value=50256)
    data = []

    if 'prompt_tags' in batch[0]:
        # Keep track of soft prompt tags
        prompt_tags = batch[0]['prompt_tags']

        for token, lenn, prompt_tag in zip(tokens_pad.T, lens, prompt_tags):
            data.append((token, lenn, tokens_to_generate, compute_logprobs, prompt_tag))
    else:
        for token, lenn in zip(tokens_pad.T, lens):
            data.append((token, lenn, tokens_to_generate, compute_logprobs))

    return data

request = []
prompt = "Translate German to English: Ich bin müde"
request.append(prompt)

tokens_to_generate = 100
compute_logprobs = True
batch_size = 8

dataset = GPTRequestDataset(request, model.tokenizer, tokens_to_generate, compute_logprobs)
request_dl = DataLoader(dataset=pad_collate(dataset), batch_size=int(batch_size))

response = trainer.predict(model, request_dl)

print(response)