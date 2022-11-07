from utils import EWC 
from bonito.multiprocessing import process_cancel
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
from bonito.reader import read_chunks, Reader

import toml
import torch
import numpy as np
from torch.utils.data import DataLoader

from torch.nn.functional import log_softmax, ctc_loss, softmax
from torch.nn import Module, ModuleList, Sequential, Conv1d, BatchNorm1d, Dropout

dirname = "/home/princezwang/software/MetsBonito/bonito/models/dna_r9.4.1@v2"
if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models__, dirname)):
    dirname = os.path.join(__models__, dirname)
pretrain_file = os.path.join(dirname, 'config.toml')
config = toml.load(pretrain_file)
if 'lr_scheduler' in config:
    print(f"[ignoring 'lr_scheduler' in --pretrained config]")
    del config['lr_scheduler']
model = load_model(dirname, "cpu")
read_path = "/home/princezwang/nanopore/correspondence/analysis/round0/m_train/filter_readids_like_c.txt"
# "/xdisk/hongxuding/ziyuan/correspondence/outputs/round0/m_6mers_1_curlcake_6_fail_ids.txt"
read_txt = open(read_path,"r")
read_ids = str(read_txt.read()).split("\n")[1:20]
read_txt.close()
reads_directory = "/home/princezwang/nanopore/dataset/dna/train_data/fast5/mod_1/set1"
reader = Reader(reads_directory, False)
reads = reader.get_reads(
    reads_directory, n_proc=8, recursive=False,
    read_ids=read_ids, skip=False,
    cancel=process_cancel()
)
reads = (
    chunk for read in reads
    for chunk in read_chunks(
        read,
        chunksize=model.config["basecaller"]["chunksize"],
        overlap=model.config["basecaller"]["overlap"]
    )
)
# reads = list(reads)
ewv = EWC(model,reads)