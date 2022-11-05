"""
Bonito Basecaller
"""

import os
import sys
import numpy as np
from tqdm import tqdm
from time import perf_counter
from functools import partial
from datetime import timedelta
from itertools import islice as take
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch

from bonito.aligner import align_map, Aligner
from bonito.reader import read_chunks, Reader
from bonito.io import CTCWriter, Writer, biofmt
from bonito.mod_util import call_mods, load_mods_model
from bonito.cli.download import File, models, __models__
from bonito.multiprocessing import process_cancel, process_itemmap
from bonito.util import column_to_set, load_symbol, load_model, init, chunk, stitch, batchify, unbatchify,permute


def main(args):

    # init(args.seed, args.device)

    try:
        reader = Reader(args.reads_directory, args.recursive)
        sys.stderr.write("> reading %s\n" % reader.fmt)
    except FileNotFoundError:
        sys.stderr.write("> error: no suitable files found in %s\n" % args.reads_directory)
        exit(1)

    try:
        dirname = "/xdisk/hongxuding/ziyuan/meta-bonito/MetsBonito/bonito/models/dna_r9.4.1@v2"
        if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models__, dirname)):
            dirname = os.path.join(__models__, dirname)
        model = load_model(dirname, "cuda")
    except FileNotFoundError:
        sys.stderr.write(f"> error: failed to load {args.model_directory}\n")
        sys.stderr.write(f"> available models:\n")
        for model in sorted(models): sys.stderr.write(f" - {model}\n")
        exit(1)

    read_path = args.read_ids
    # "/xdisk/hongxuding/ziyuan/correspondence/outputs/round0/m_6mers_1_curlcake_6_fail_ids.txt"
    read_txt = open(read_path,"r")
    read_ids = str(read_txt.read()).split("\n")
    read_txt.close()

    reads = reader.get_reads(
        args.reads_directory, n_proc=8, recursive=args.recursive,
        read_ids=read_ids, skip=args.skip,
        cancel=process_cancel()
    )
    embs = extract_embeddings(model,reads,chunksize=10000)
    embs = list(embs)
    embs = [embs[i][1]["embeddings"][0:6000].numpy() for i in range(len(embs))]
    embs = np.array(embs[0:50])
    np.savetxt(args.output_file, embs.reshape(embs.shape[0],-1), delimiter = " ", fmt="%.3e")
    # print(embs[1])

def extract_embeddings(model, reads, chunksize=0):
    """
    Basecalls a set of reads.
    """
    chunks = (
        (read, chunk(torch.tensor(read.signal), chunksize, 0)) for read in reads
    )
    embeddings = unbatchify(
        (k, compute_embedding(model, v)) for k, v in batchify(chunks, 8)
    )
    embeddings = (
        (read, {'embeddings': stitch(v, chunksize, 0, len(read.signal), model.stride)}) for read, v in embeddings
    )

    return embeddings

def compute_embedding(model, batch):
    """
    Compute embedding for model.
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        chunks = batch.to(device)
        probs = permute(model.encoder(chunks),'TNC', 'TCN')
    return probs.cpu().to(torch.float32)

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("reads_directory")
    parser.add_argument("--modified-bases", nargs="+")
    parser.add_argument("output_file")
    parser.add_argument("--modified-procs", default=8, type=int)
    parser.add_argument("--modified-device", default=None)
    parser.add_argument("read_ids")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--weights", default=0, type=int)
    parser.add_argument("--skip", action="store_true", default=False)
    parser.add_argument("--save-ctc", action="store_true", default=False)
    parser.add_argument("--revcomp", action="store_true", default=False)
    parser.add_argument("--recursive", action="store_true", default=False)
    quant_parser = parser.add_mutually_exclusive_group(required=False)
    quant_parser.add_argument("--quantize", dest="quantize", action="store_true")
    quant_parser.add_argument("--no-quantize", dest="quantize", action="store_false")
    parser.set_defaults(quantize=None)
    parser.add_argument("--overlap", default=None, type=int)
    parser.add_argument("--chunksize", default=None, type=int)
    parser.add_argument("--batchsize", default=None, type=int)
    parser.add_argument("--max-reads", default=10, type=int)
    parser.add_argument("--alignment-threads", default=8, type=int)
    parser.add_argument('-v', '--verbose', action='count', default=0)
    return parser

main(argparser().parse_args())