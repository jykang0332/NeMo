# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""
# Preparing the Tokenizer for the dataset
Use the `process_asr_text_tokenizer.py` script under <NEMO_ROOT>/scripts/tokenizers/ in order to prepare the tokenizer.

```sh
python <NEMO_ROOT>/scripts/tokenizers/process_asr_text_tokenizer.py \
        --manifest=<path to train manifest files, seperated by commas>
        OR
        --data_file=<path to text data, seperated by commas> \
        --data_root="<output directory>" \
        --vocab_size=<number of tokens in vocabulary> \
        --tokenizer=<"spe" or "wpe"> \
        --no_lower_case \
        --spe_type=<"unigram", "bpe", "char" or "word"> \
        --spe_character_coverage=1.0 \
        --log
```

# Training the model
```sh
python speech_to_text_ctc_bpe.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.validation_ds.manifest_filepath=<path to val/test manifest> \
    model.tokenizer.dir=<path to directory of tokenizer (not full path to the vocab file!)> \
    model.tokenizer.type=<either bpe or wpe> \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.strategy="ddp" \
    trainer.max_epochs=100 \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_steps=2000
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="<Name of experiment>" \
    exp_manager.wandb_logger_kwargs.project="<Name of project>"
```

# Fine-tune a model

For documentation on fine-tuning this model, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html#fine-tuning-configurations

# Pretrained Models

For documentation on existing pretrained models, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/results.html

"""

import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models.ctc_bpe_models_Fitnet_AddLayer import EncDecCTCModelBPE
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

import torch
# weight selection
def uniform_element_selection(wt, s_shape):
    assert wt.dim() == len(s_shape), "Tensors have different number of dimensions"
    ws = wt.clone()
    for dim in range(wt.dim()):
        assert wt.shape[dim] >= s_shape[dim], "Teacher's dimension should not be smaller than student's dimension"  # determine whether teacher is larger than student on this dimension
        if wt.shape[dim] % s_shape[dim] == 0:
            step = wt.shape[dim] // s_shape[dim]
            indices = torch.arange(s_shape[dim]) * step
        else:
            indices = torch.round(torch.linspace(0, wt.shape[dim], s_shape[dim]))
        if indices[-1] >= wt.shape[dim]:
            indices[-1] = wt.shape[dim] - 1
        ws = torch.index_select(ws, dim, indices.long().cuda())
    assert ws.shape == s_shape
    return ws


@hydra_runner(config_path="./conf", config_name="conformer_ctc_bpe_Fitnet_AddLayer")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecCTCModelBPE(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    # teacher decoder load
    te_dec_bias = torch.load('/data/jykang/NeMo/data/decoder/te_dec_bias.pt')
    te_dec_weight = torch.load('/data/jykang/NeMo/data/decoder/te_dec_weight.pt')
    te_enc = torch.load('/data/jykang/NeMo/data/encoder/enc_weights.pt')
    te_preprocessor = torch.load('/data/jykang/NeMo/data/encoder/preprocessor_weights.pt')

    with torch.no_grad():
        asr_model.state_dict()['decoder.decoder_layers.0.bias'].copy_(te_dec_bias)
        asr_model.state_dict()['decoder.decoder_layers.0.weight'].copy_(te_dec_weight)
        for key, value in asr_model.state_dict().items():
            if 'preprocessor' in key:
                asr_model.state_dict()[key].copy_(te_preprocessor[key])

            if 'encoder' in key:
                if 'layers' in key:
                    # layer change (+2)
                    part = key.split('.')
                    layer_num = int(part[2]) + 2
                    part[2] = str(layer_num)
                    new_key = '.'.join(part)
                    weight_selection = uniform_element_selection(te_enc[new_key], value.shape)
                else:
                    weight_selection = uniform_element_selection(te_enc[key], value.shape)
                    
                asr_model.state_dict()[key].copy_(weight_selection)

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()
