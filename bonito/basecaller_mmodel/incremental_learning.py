"""
Bonito Model template
"""

from pickletools import optimize
import numpy as np
from random import shuffle

from torch import  nn, optim
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
from torch.utils.data import DataLoader, TensorDataset
from bonito.reader import Reader


from train_flipflop import gen_batch
from _bin_argparse import get_train_flipflop_parser
from early_stop import EarlyStopping

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


def train_one_epoch(model,batch_list):
    loss_total = {'total_loss': 0, 'loss': 0, 'label_smooth_loss': 0, "Regularization_loss" : 0}
    for train_batch_tensor,train_seqref,train_seqlen,_,_,_ in batch_list:
        scores = model(permute(train_batch_tensor,[0,1,2],[1,2,0]) )
        loss = model.ctc_label_smoothing_loss(scores, train_seqref, train_seqlen)
        loss_total ['total_loss'] += loss['total_loss']
        loss_total ['label_smooth_loss'] += loss['label_smooth_loss']
        loss_total ['loss'] += loss['loss']
        loss_total ['Regularization_loss'] += loss['Regularization_loss']
    return {k:v/len(batch_list) for k,v in loss_total.items()}

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

# for p in model.encoder.parameters():
#     p.requires_grad=False

optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=5e-4)

def generate_dataset(dir):
    main_batch = gen_batch(dir = dir)
    batch_list = list(main_batch)
    batch_tensor, seqref, seqlen = batch_list[0][0],batch_list[0][1],batch_list[0][2]
    return batch_tensor, seqref, seqlen

losses,classification_losses,reg_loss,test_losses,canonical_losses,validation_losses = [],[],[],[],[],[]

merge_list = []
for i in range(20):
    batch_list = gen_batch(dir = "/home/princezwang/nanopore/dataset/dna/train_data/hdf5/mod_1/set1/batch"+str(i)+".hdf5")
    merge_list += list(batch_list)

shuffle(merge_list)
train_list,validation_list = merge_list[0:int(len(merge_list)*0.7)],merge_list[int(len(merge_list)*0.7):]

# train_batch_tensor,train_seqref,train_seqlen = generate_dataset("/home/princezwang/nanopore/dataset/dna/train_data/hdf5/mod_1/set1/batch0.hdf5")
test_batch_tensor,test_seqref,test_seqlen = generate_dataset("/home/princezwang/nanopore/dataset/dna/train_data/hdf5/mod_1/set1/batch30.hdf5")
canonical_batch_tensor,canonical_seqref,canonical_seqlen = generate_dataset("/home/princezwang/nanopore/dataset/dna/train_data/hdf5/canonical/set1/batch0.hdf5")
early_stopping = EarlyStopping(patience=15, verbose=True,path="/home/princezwang/nanopore/results/dna/meta_bonito_dna.pt")
for i in range(10000):
    optimizer.zero_grad()
    # validation_batch_tensor,validation_seqref,validation_seqlen = generate_dataset("/home/princezwang/nanopore/dataset/dna/train_data/hdf5/mod_1/set1/batch0.hdf5")

    loss = train_one_epoch(model,train_list)
    validation_loss = train_one_epoch(model,validation_list)

    test_scores = model(permute(test_batch_tensor,[0,1,2],[1,2,0]) )
    test_loss = model.ctc_label_smoothing_loss(test_scores, test_seqref, test_seqlen)
    canonical_scores = model(permute(canonical_batch_tensor,[0,1,2],[1,2,0]) )
    canonical_loss = model.ctc_label_smoothing_loss(canonical_scores, canonical_seqref, canonical_seqlen)
    # validation_scores = model(permute(validation_batch_tensor,[0,1,2],[1,2,0]) )
    # validation_loss = model.ctc_label_smoothing_loss(validation_scores, validation_seqref, validation_seqlen)

    loss['total_loss'].backward()
    nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=2, norm_type=2)
    optimizer.step()


    losses.append(float(loss['total_loss']))
    classification_losses.append(float(loss['loss']))
    reg_loss.append(float(test_loss['Regularization_loss']))
    test_losses.append(float(test_loss['loss']))
    canonical_losses.append(float(canonical_loss['loss']))
    validation_losses.append(float(validation_loss['loss']))
    

    early_stopping(validation_loss['total_loss'], model)
    if early_stopping.early_stop:
            break

np.savetxt("/home/princezwang/nanopore/results/dna/losses.csv.gz", np.array(losses) , delimiter=" ",fmt='%.3e')
np.savetxt("/home/princezwang/nanopore/results/dna/class_losses.csv.gz", np.array(classification_losses) , delimiter=" ",fmt='%.3e')
np.savetxt("/home/princezwang/nanopore/results/dna/r1_losses.csv.gz", np.array(reg_loss) , delimiter=" ",fmt='%.3e')
np.savetxt("/home/princezwang/nanopore/results/dna/test_losses.csv.gz", np.array(test_losses) , delimiter=" ",fmt='%.3e')
np.savetxt("/home/princezwang/nanopore/results/dna/canonical_losses.csv.gz", np.array(canonical_losses) , delimiter=" ",fmt='%.3e')
np.savetxt("/home/princezwang/nanopore/results/dna/validation_losses.csv.gz", np.array(validation_losses) , delimiter=" ",fmt='%.3e')

model



        