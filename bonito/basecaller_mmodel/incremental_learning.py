"""
Bonito Model template
"""

import numpy as np

from bonito.nn import Permute, layers
import torch
from torch.nn.functional import log_softmax, ctc_loss
from torch.nn import Module, ModuleList, Sequential, Conv1d, BatchNorm1d, Dropout
import toml
import torch
import numpy as np
import os

from bonito.util import __models__, default_config, default_data, load_model
from bonito.util import  load_symbol, init, half_supported, permute
from bonito.training import load_state, Trainer
from fast_ctc_decode import beam_search, viterbi_search

class Decoder(Module):
    """
    Decoder
    """
    def __init__(self, features, classes):
        super(Decoder, self).__init__()
        self.layers = Sequential(
            Conv1d(features, classes, kernel_size=1, bias=True),
            Permute([2, 0, 1])
        )

    def forward(self, x):
        return log_softmax(self.layers(x), dim=-1)

def decode_ref(encoded, labels):
    """
    Convert a integer encoded reference into a string and remove blanks
    """
    return ''.join(labels[e] for e in encoded.tolist() if e)

dirname = "/home/princezwang/software/bonito/bonito/models/dna_r9.4.1@v2"
if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models__, dirname)):
    dirname = os.path.join(__models__, dirname)
pretrain_file = os.path.join(dirname, 'config.toml')
config = toml.load(pretrain_file)
if 'lr_scheduler' in config:
    print(f"[ignoring 'lr_scheduler' in --pretrained config]")
    del config['lr_scheduler']
model = load_model(dirname, 'cpu')
# model = load_symbol(config, 'Model')(config)
# scores = model(torch.randn(260,1,1000))
# seqs = [model.decode(x) for x in permute(scores, 'TNC', 'NTC')]
decoder_ww = model.state_dict()["decoder.layers.0.weight"]
decoder_bias = model.state_dict()["decoder.layers.0.bias"]
decoder_ww = torch.cat([decoder_ww,decoder_ww[2].reshape(-1,48,1)])
decoder_bias = torch.cat([decoder_bias,decoder_bias[2].reshape(-1)],0)
decoder_m = Decoder(48,6)
param_new = {}
for key in decoder_m.state_dict().keys():
    param_new[key] = decoder_m.state_dict()[key]
    print(key, decoder_m.state_dict()[key].shape)
param_new['layers.0.weight'] = decoder_ww
param_new['layers.0.bias'] = decoder_bias
decoder_m.load_state_dict(param_new)
model.decoder = decoder_m
model.alphabet = ['N', 'A', 'C', 'G', 'T', 'M']
# model = load_symbol(config, 'Model')(config)
scores = model(torch.randn(260,1,1000))
seqs = [model.decode(x) for x in permute(scores, 'TNC', 'NTC')]
print(seqs)