#!/usr/bin/env python3

"""
Bonito training.
"""

import os
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from importlib import import_module

from bonito.data import load_numpy, load_script
from bonito.util import __models__, default_config, default_data
from bonito.util import load_model, load_symbol, init, half_supported
from bonito.training import load_state, Trainer
from bonito.nn import Permute, layers

import toml
import torch
import numpy as np
from torch.utils.data import DataLoader

from torch.nn.functional import log_softmax, ctc_loss, softmax
from torch.nn import Module, ModuleList, Sequential, Conv1d, BatchNorm1d, Dropout

class Decoder(Module):
    """
    Decoder
    """
    def __init__(self, features, classes,mod_classes = 1):
        super(Decoder, self).__init__()
        self.layers_origin = Sequential(
            Conv1d(features, classes, kernel_size=1, bias=True),
            Permute([2, 0, 1])
        )
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
        return log_softmax(torch.cat((torch.cat((canonical_z[:,:,0:3],mod_z),2),canonical_z[:,:,3:]),2), dim =-1)

    def calculate_kdloss(self, x):
        canonical_z = softmax(self.layers_origin(x)/2, dim=2)
        original = softmax(self.layers(x)/2, dim=2)
        kd_loss= -torch.sum(original.mul(torch.log(canonical_z)))
        return kd_loss/x.shape[0]


def main(args):

    workdir = os.path.expanduser(args.training_directory)

    if os.path.exists(workdir) and not args.force:
        print("[error] %s exists, use -f to force continue training." % workdir)
        exit(1)

    init(args.seed, args.device, (not args.nondeterministic))
    device = torch.device(args.device)

    print("[loading data]")
    try:
        train_loader_kwargs, valid_loader_kwargs = load_numpy(
            args.chunks, args.directory
        )
    except FileNotFoundError:
        train_loader_kwargs, valid_loader_kwargs = load_script(
            args.directory,
            seed=args.seed,
            chunks=args.chunks,
            valid_chunks=args.valid_chunks
        )

    loader_kwargs = {
        "batch_size": args.batch, "num_workers": 4, "pin_memory": True
    }
    train_loader = DataLoader(**loader_kwargs, **train_loader_kwargs)
    valid_loader = DataLoader(**loader_kwargs, **valid_loader_kwargs)

    if not args.pretrained:
        config = toml.load(args.config)
    else:
        dirname = args.pretrained
        if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models__, dirname)):
            dirname = os.path.join(__models__, dirname)
        pretrain_file = os.path.join(dirname, 'config.toml')
        config = toml.load(pretrain_file)
        if 'lr_scheduler' in config:
            print(f"[ignoring 'lr_scheduler' in --pretrained config]")
            del config['lr_scheduler']

    argsdict = dict(training=vars(args))

    os.makedirs(workdir, exist_ok=True)
    toml.dump({**config, **argsdict}, open(os.path.join(workdir, 'config.toml'), 'w'))

    # print("[loading model]")
    # if args.pretrained:
    #     print("[using pretrained model {}]".format(args.pretrained))
    #     model = load_model(args.pretrained, device, half=False)
    # else:
    #     model = load_symbol(config, 'Model')(config)
    dirname = "/xdisk/hongxuding/ziyuan/meta-bonito/MetsBonito/bonito/models/dna_r9.4.1@v2"
    if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models__, dirname)):
        dirname = os.path.join(__models__, dirname)
    pretrain_file = os.path.join(dirname, 'config.toml')
    config = toml.load(pretrain_file)
    if 'lr_scheduler' in config:
        print(f"[ignoring 'lr_scheduler' in --pretrained config]")
        del config['lr_scheduler']
    model = load_model(dirname, "cuda")
    
    decoder_ww = model.state_dict()["decoder.layers.0.weight"]
    decoder_bias = model.state_dict()["decoder.layers.0.bias"]

    decoder_m = Decoder(48,5, mod_classes = 2)
    param_new = {}
    for key in decoder_m.state_dict().keys():
        param_new[key] = decoder_m.state_dict()[key]
        print(key, decoder_m.state_dict()[key].shape)
    param_new['layers.0.weight'] = decoder_ww
    param_new['layers.0.bias'] = decoder_bias
    param_new['layers_origin.0.weight'] = decoder_ww
    param_new['layers_origin.0.bias'] = decoder_bias
    # param_new['layers_mod.0.weight'] = decoder_ww[2].reshape([1,48,1])
    # param_new['layers_mod.0.bias'] = decoder_bias[2].reshape(-1)

    decoder_m.load_state_dict(param_new)
    model.decoder = decoder_m
    model.alphabet = ['N', 'A', 'C', 'm','h','G', 'T']
    torch.save(model,os.path.join(workdir, "optim_%s.pt" % 0))
    model = torch.load("/xdisk/hongxuding/ziyuan/meta-bonito/results/dna/step_step/optim_3.pt")
    if config.get("lr_scheduler"):
        sched_config = config["lr_scheduler"]
        lr_scheduler_fn = getattr(
            import_module(sched_config["package"]), sched_config["symbol"]
        )(**sched_config)
    else:
        lr_scheduler_fn = None

    trainer = Trainer(
        model, device, train_loader, valid_loader,
        use_amp=half_supported() and not args.no_amp,
        lr_scheduler_fn=lr_scheduler_fn,
        restore_optim=args.restore_optim,
        save_optim_every=args.save_optim_every,
        grad_accum_split=args.grad_accum_split
    )

    if (',' in args.lr):
        lr = [float(x) for x in args.lr.split(',')]
    else:
        lr = float(args.lr)
    trainer.fit(workdir, args.epochs, lr)

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("training_directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--config', default=default_config)
    group.add_argument('--pretrained', default="")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", default='2e-6')
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--chunks", default=0, type=int)
    parser.add_argument("--valid-chunks", default=1000, type=int)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("-f", "--force", action="store_true", default=True)
    parser.add_argument("--restore-optim", action="store_true", default=False)
    parser.add_argument("--nondeterministic", action="store_true", default=False)
    parser.add_argument("--save-optim-every", default=1, type=int)
    parser.add_argument("--grad-accum-split", default=1, type=int)
    return parser

# args_parser = argparser().parse_args()
# # # args_parser.training_directory = "/xdisk/hongxuding/ziyuan/meta-bonito/results/dna/canonical"
# args_parser.directory = "/xdisk/hongxuding/ziyuan/meta-bonito/results/dataset/npy"
# args_parser.epochs = 15
# args_parser.lr = "1e-5" 
# # args_parser.chunks = 3000

# main(args_parser)

