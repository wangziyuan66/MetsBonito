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
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


from train_flipflop import gen_batch
from _bin_argparse import get_train_flipflop_parser
from early_stop import EarlyStopping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#######################################################################
##########                   Device Import                   ##########
#######################################################################

class Decoder(Module):
    """
    Decoder
    """
    def __init__(self, features, classes,mod_classes = 1):
        super(Decoder, self).__init__()
        self.layers = Sequential(
            Conv1d(features, classes, kernel_size=1, bias=True),
            Permute([2, 0, 1])
        )
        self.layers_mod = Sequential(
            Conv1d(features, mod_classes, kernel_size=1, bias=True),
            Permute([2, 0, 1])
        )

    def forward(self, x):
        canonical_z = self.layers(x)
        mod_z = self.layers_mod(x)
        return log_softmax(torch.cat((canonical_z,mod_z),2), dim=-1)


def decode_ref(encoded, labels):
    """
    Convert a integer encoded reference into a string and remove blanks
    """
    return ''.join(labels[e] for e in encoded.tolist() if e)


def train_one_epoch(model,batch_list):
    model.to(device)
    loss_total = {'total_loss': 0, 'loss': 0, 'label_smooth_loss': 0, "Regularization_loss" : 0}
    for train_batch_tensor,train_seqref,train_seqlen,_,_,_ in batch_list:
        train_batch_tensor,train_seqref,train_seqlen = train_batch_tensor.to(device),train_seqref.to(device),train_seqlen.to(device)
        scores = model(permute(train_batch_tensor,[0,1,2],[1,2,0]) )
        loss = model.ctc_label_smoothing_loss(scores, train_seqref, train_seqlen)
        seq = model.decode(scores)
        loss_total ['total_loss'] += loss['total_loss']
        loss_total ['label_smooth_loss'] += loss['label_smooth_loss']
        loss_total ['loss'] += loss['loss']
        loss_total ['Regularization_loss'] += loss['Regularization_loss']
    return {k:v/len(batch_list) for k,v in loss_total.items()}

def validation_one_epoch(model,batch_list):
    model.to(device)
    loss_total = {'total_loss': 0, 'loss': 0, 'label_smooth_loss': 0, "Regularization_loss" : 0}
    with torch.no_grad():
        for train_batch_tensor,train_seqref,train_seqlen,_,_,_ in batch_list:
            train_batch_tensor,train_seqref,train_seqlen = train_batch_tensor.to(device),train_seqref.to(device),train_seqlen.to(device)
            scores = model(permute(train_batch_tensor,[0,1,2],[1,2,0]) )
            loss = model.ctc_label_smoothing_loss(scores, train_seqref, train_seqlen, weights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
            loss_total ['total_loss'] += loss['total_loss']
            loss_total ['label_smooth_loss'] += loss['label_smooth_loss']
            loss_total ['loss'] += loss['loss']
            loss_total ['Regularization_loss'] += loss['Regularization_loss']
    return {k:v/len(batch_list) for k,v in loss_total.items()}

def generate_dataset(dir):
    main_batch = gen_batch(dir = dir)
    batch_list = list(main_batch)
    batch_tensor, seqref, seqlen = batch_list[0][0],batch_list[0][1],batch_list[0][2]
    return batch_tensor, seqref, seqlen

<<<<<<< HEAD
def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("--checkpoint", default=None)
    return parser

def main(args):

    if args.checkpoint==None:
        dirname = "/xdisk/hongxuding/ziyuan/meta-bonito/MetsBonito/bonito/models/dna_r9.4.1@v2"
        if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models__, dirname)):
            dirname = os.path.join(__models__, dirname)
        pretrain_file = os.path.join(dirname, 'config.toml')
        config = toml.load(pretrain_file)
        if 'lr_scheduler' in config:
            print(f"[ignoring 'lr_scheduler' in --pretrained config]")
            del config['lr_scheduler']
        model = load_model(dirname, "cpu")
        decoder_ww = model.state_dict()["decoder.layers.0.weight"]
        decoder_bias = model.state_dict()["decoder.layers.0.bias"]

        decoder_m = Decoder(48,5)
        param_new = {}
        for key in decoder_m.state_dict().keys():
            param_new[key] = decoder_m.state_dict()[key]
            print(key, decoder_m.state_dict()[key].shape)
        param_new['layers.0.weight'] = decoder_ww
        param_new['layers.0.bias'] = decoder_bias
        decoder_m.load_state_dict(param_new)
        model.decoder = decoder_m
        model.alphabet = ['N', 'A', 'C', 'G', 'T', 'm']
        # for p in model.encoder.parameters():
        #     p.requires_grad=False
#######################################################################
##########                  From Beginning                   ##########
#######################################################################
    if args.checkpoint!=None:
        model = torch.load(args.checkpoint)
#######################################################################
##########                 From Checkpoint                   ##########
#######################################################################

    optimizer = optim.SGD(model.decoder.layers_mod.parameters(), lr=1e-1)


    losses,classification_losses,reg_loss,test_losses,canonical_losses,validation_losses = [],[],[],[],[],[]

    merge_list = []
    i = 11
    batch_list = gen_batch(dir = "/xdisk/hongxuding/dinglab/nanopore_working_group/dna/train_data/hdf5/mod_1/set1/batch"+str(i)+".hdf5")
=======
losses,classification_losses,reg_loss,test_losses,canonical_losses,validation_losses = [],[],[],[],[],[]

merge_list = []
for i in range(1):
    batch_list = gen_batch(dir = "/home/princezwang/nanopore/dataset/dna/train_data/hdf5/mod_1/set1/batch"+str(i)+".hdf5")
>>>>>>> afe0c8c0a7e5d542347a053af710b8e923ed8238
    merge_list += list(batch_list)

    shuffle(merge_list)
    train_list,validation_list = merge_list[0:int(len(merge_list)*0.7)],merge_list[int(len(merge_list)*0.7):]

    # train_batch_tensor,train_seqref,train_seqlen = generate_dataset("/xdisk/hongxuding/dinglab/nanopore_working_group/dna/train_data/hdf5/mod_1/set1/batch0.hdf5")
    test_batch_tensor,test_seqref,test_seqlen = generate_dataset("/xdisk/hongxuding/dinglab/nanopore_working_group/dna/train_data/hdf5/mod_1/set1/batch30.hdf5")
    canonical_batch_tensor,canonical_seqref,canonical_seqlen = generate_dataset("/xdisk/hongxuding/dinglab/nanopore_working_group/dna/train_data/hdf5/canonical/set1/batch0.hdf5")
    early_stopping = EarlyStopping(patience=15, verbose=True,path="/xdisk/hongxuding/ziyuan/meta-bonito/results/dna/meta_bonito_dna.pt")
    pt_path = "/xdisk/hongxuding/ziyuan/meta-bonito/results/dna/meta_bonito_dna.pt"
    for i in range(5000):
        optimizer.zero_grad()
        # validation_batch_tensor,validation_seqref,validation_seqlen = generate_dataset("/home/princezwang/nanopore/dataset/dna/train_data/hdf5/mod_1/set1/batch0.hdf5")

        loss = train_one_epoch(model,train_list)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        validation_loss = validation_one_epoch(model,validation_list)
        print(float(validation_loss['Regularization_loss']))
        test_scores = model(permute(test_batch_tensor.to(device),[0,1,2],[1,2,0]) )
        test_loss = model.ctc_label_smoothing_loss(test_scores, test_seqref.to(device), test_seqlen.to(device))
        canonical_scores = model(permute(canonical_batch_tensor.to(device),[0,1,2],[1,2,0]) )
        canonical_loss = model.ctc_label_smoothing_loss(canonical_scores, canonical_seqref.to(device), canonical_seqlen.to(device))
        # validation_scores = model(permute(validation_batch_tensor,[0,1,2],[1,2,0]) )
        # validation_loss = model.ctc_label_smoothing_loss(validation_scores, validation_seqref, validation_seqlen)

        loss['total_loss'].backward()
        # nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=2, norm_type=2)
        optimizer.step()


        losses.append(float(loss['total_loss']))
        classification_losses.append(float(loss['loss']))
        reg_loss.append(float(test_loss['Regularization_loss']))
        test_losses.append(float(test_loss['loss']))
        canonical_losses.append(float(canonical_loss['loss']))
        validation_losses.append(float(validation_loss['loss']))
        
        if i%100==0:
            early_stopping.path = pt_path.split(".pt")[0]+"_"+str(i)+"epoch.pt"
            early_stopping(validation_loss['total_loss'], model)
            if early_stopping.early_stop:
                    break


    np.savetxt("/xdisk/hongxuding/ziyuan/meta-bonito/results/dna/losses.csv.gz", np.array(losses) , delimiter=" ",fmt='%.3e')
    np.savetxt("/xdisk/hongxuding/ziyuan/meta-bonito/results/dna/class_losses.csv.gz", np.array(classification_losses) , delimiter=" ",fmt='%.3e')
    np.savetxt("/xdisk/hongxuding/ziyuan/meta-bonito/results/dna/r1_losses.csv.gz", np.array(reg_loss) , delimiter=" ",fmt='%.3e')
    np.savetxt("/xdisk/hongxuding/ziyuan/meta-bonito/results/dna/test_losses.csv.gz", np.array(test_losses) , delimiter=" ",fmt='%.3e')
    np.savetxt("/xdisk/hongxuding/ziyuan/meta-bonito/results/dna/canonical_losses.csv.gz", np.array(canonical_losses) , delimiter=" ",fmt='%.3e')
    np.savetxt("/xdisk/hongxuding/ziyuan/meta-bonito/results/dna/validation_losses.csv.gz", np.array(validation_losses) , delimiter=" ",fmt= '%.3e')

    model


main(argparser().parse_args())
