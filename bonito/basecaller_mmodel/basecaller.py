"""
Bonito Basecaller
"""

from asyncio.trsock import TransportSocket
import os
import sys
import numpy as np
from tqdm import tqdm
from time import perf_counter
from functools import partial
from datetime import timedelta
from itertools import islice as take
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import array
import re

from bonito.aligner import align_map, Aligner
from bonito.reader import read_chunks, Reader
from bonito.io import CTCWriter, Writer, biofmt
from bonito.mod_util import call_mods, load_mods_model
from bonito.cli.download import File, models, __models__
from bonito.multiprocessing import process_cancel, process_itemmap
from bonito.util import column_to_set, load_symbol, load_model, init
from bonito.ctc import basecall

import torch
from torch import  nn, optim
from bonito.nn import Permute, layers
import torch
from torch.nn.functional import log_softmax, ctc_loss
from torch.nn import Module, ModuleList, Sequential, Conv1d, BatchNorm1d, Dropout


from remora.util import format_mm_ml_tags,softmax_axis1

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

def main(args):

    # init(args.seed, args.device)

    try:
        reader = Reader(args.reads_directory, args.recursive)
        sys.stderr.write("> reading %s\n" % reader.fmt)
    except FileNotFoundError:
        sys.stderr.write("> error: no suitable files found in %s\n" % args.reads_directory)
        exit(1)

    fmt = biofmt(aligned=args.reference is not None)

    if args.reference and args.reference.endswith(".mmi") and fmt.name == "cram":
        sys.stderr.write("> error: reference cannot be a .mmi when outputting cram\n")
        exit(1)
    elif args.reference and fmt.name == "fastq":
        sys.stderr.write(f"> warning: did you really want {fmt.aligned} {fmt.name}?\n")
    else:
        sys.stderr.write(f"> outputting {fmt.aligned} {fmt.name}\n")

    if args.model_directory in models and args.model_directory not in os.listdir(__models__):
        sys.stderr.write("> downloading model\n")
        File(__models__, models[args.model_directory]).download()

    sys.stderr.write(f"> loading model {args.model_directory}\n")
    try:
        model = torch.load(args.model_directory,map_location=torch.device('cpu'))
    except FileNotFoundError:
        sys.stderr.write(f"> error: failed to load {args.model_directory}\n")
        sys.stderr.write(f"> available models:\n")
        for model in sorted(models): sys.stderr.write(f" - {model}\n")
        exit(1)

    if args.verbose:
        sys.stderr.write(f"> model basecaller params: {model.config['basecaller']}\n")

    # basecall = load_symbol(args.model_directory, "basecall")

    mods_model = None
    # if args.modified_base_model is not None or args.modified_bases is not None:
    #     sys.stderr.write("> loading modified base model\n")
    #     mods_model = load_mods_model(
    #         args.modified_bases, args.model_directory, args.modified_base_model,
    #         device=args.modified_device,
    #     )
    #     sys.stderr.write(f"> {mods_model[1]['alphabet_str']}\n")

    if args.reference:
        sys.stderr.write("> loading reference\n")
        aligner = Aligner(args.reference, preset='map-ont', best_n=1)
        if not aligner:
            sys.stderr.write("> failed to load/build index\n")
            exit(1)
    else:
        aligner = None

    if args.save_ctc and not args.reference:
        sys.stderr.write("> a reference is needed to output ctc training data\n")
        exit(1)

    if fmt.name != 'fastq':
        groups, num_reads = reader.get_read_groups(
            args.reads_directory, args.model_directory,
            n_proc=8, recursive=args.recursive,
            read_ids=column_to_set(args.read_ids), skip=args.skip,
            cancel=process_cancel()
        )
    else:
        groups = []
        num_reads = None

    reads = reader.get_reads(
        args.reads_directory, n_proc=8, recursive=args.recursive,
        read_ids=column_to_set(args.read_ids), skip=args.skip,
        cancel=process_cancel()
    )

    if args.max_reads:
        reads = take(reads, args.max_reads)

    if args.save_ctc:
        reads = (
            chunk for read in reads
            for chunk in read_chunks(
                read,
                chunksize=model.config["basecaller"]["chunksize"],
                overlap=model.config["basecaller"]["overlap"]
            )
        )
        ResultsWriter = CTCWriter
    else:
        ResultsWriter = CTCWriter
    

    results = basecall(
        model, reads, reverse=args.revcomp,
        batchsize=model.config["basecaller"]["batchsize"],
        chunksize=model.config["basecaller"]["chunksize"],
        overlap=model.config["basecaller"]["overlap"],
    )

    results = ((k,extend_mm_ml_tag(k,v))for k,v in results)
    # if mods_model is not None:
    #     if args.modified_device:
    #         results = ((k, call_mods(mods_model, k, v)) for k, v in results)
    #     else:
    #         results = process_itemmap(
    #             partial(call_mods, mods_model), results, n_proc=args.modified_procs
    #         )
    if aligner:
        results = align_map(aligner, results, n_thread=args.alignment_threads)

    writer = ResultsWriter(
        fmt.mode, tqdm(results, desc="> calling", unit=" reads", leave=False,
                       total=num_reads, smoothing=0, ascii=True, ncols=100),
        aligner=aligner, group_key=args.model_directory,
        ref_fn=args.reference, groups=groups,
    )

    t0 = perf_counter()
    writer.start()
    writer.join()
    duration = perf_counter() - t0
    num_samples = sum(num_samples for read_id, num_samples in writer.log)

    sys.stderr.write("> completed reads: %s\n" % len(writer.log))
    sys.stderr.write("> duration: %s\n" % timedelta(seconds=np.round(duration)))
    sys.stderr.write("> samples per second %.1E\n" % (num_samples / duration))
    sys.stderr.write("> done\n")


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("reads_directory")
    parser.add_argument("--reference")
    parser.add_argument("--modified-bases", nargs="+")
    parser.add_argument("--modified-base-model")
    parser.add_argument("--modified-procs", default=8, type=int)
    parser.add_argument("--modified-device", default=None)
    parser.add_argument("--read-ids")
    parser.add_argument("--device", default="cuda")
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
    parser.add_argument("--chunksize", default=2000, type=int)
    parser.add_argument("--batchsize", default=None, type=int)
    parser.add_argument("--max-reads", default=100, type=int)
    parser.add_argument("--alignment-threads", default=8, type=int)
    parser.add_argument('-v', '--verbose', action='count', default=0)
    return parser

def extend_mm_ml_tag(read,read_attrs,model = None):
    # scores = model(torch.tensor(read.signal).reshape(-1,1,1))
    # probs = softmax_axis1(scores.reshape(-1,scores.shape[2]).detach().numpy())[:, 1:].astype(np.float64)
    mod_idx = [substr.start() for substr in re.finditer("m" , read_attrs['sequence'])]
    mm,ml = format_mm_tags(read_attrs['sequence'],mod_idx,"m","C")
    
    read_attrs['mods'] =[
        f"Mm:Z:{mm}",
        f"Ml:B:C,{','.join(map(str, ml))}"
    ]
    return read_attrs 

def format_mm_ml_tags(seq, poss, probs, mod_bases, can_base):
    """Format MM and ML tags for BAM output. See
    https://samtools.github.io/hts-specs/SAMtags.pdf for format details.

    Args:
        seq (str): read-centric read sequence. For reference-anchored calls
            this should be the reverse complement sequence.
        poss (list): positions relative to seq
        probs (np.array): probabilties for modified bases
        mod_bases (str): modified base single letter codes
        can_base (str): canonical base

    Returns:
        MM string tag and ML array tag
    """

    # initialize dict with all called mods to make sure all called mods are
    # shown in resulting tags
    per_mod_probs = dict((mod_base, []) for mod_base in mod_bases)
    for pos, mod_probs in sorted(zip(poss, probs)):
        # mod_lps is set to None if invalid sequence is encountered or too
        # few events are found around a mod
        if mod_probs is None:
            continue
        for mod_prob, mod_base in zip(mod_probs, mod_bases):
            mod_prob = mod_prob
            per_mod_probs[mod_base].append((pos, mod_prob))

    mm_tag, ml_tag = "", array.array("B")
    for mod_base, pos_probs in per_mod_probs.items():
        if len(pos_probs) == 0:
            continue
        mod_poss, probs = zip(*sorted(pos_probs))
        # compute modified base positions relative to the running total of the
        # associated canonical base
        can_base_mod_poss = (
            np.cumsum([1 if b == can_base else 0 for b in seq])[
                np.array(mod_poss)
            ]
            - 1
        )
        mod_gaps = ",".join(
            map(str, np.diff(np.insert(can_base_mod_poss, 0, -1)) - 1)
        )
        mm_tag += f"{can_base}+{mod_base}?,{mod_gaps};"
        # extract mod scores and scale to 0-255 range
        scaled_probs = np.floor(np.array(probs) * 256)
        # last interval includes prob=1
        scaled_probs[scaled_probs == 256] = 255
        ml_tag.extend(scaled_probs.astype(np.uint8))

    return mm_tag, ml_tag


def format_mm_tags(seq, poss, mod_base, can_base):
    """Format MM and ML tags for BAM output. See
    https://samtools.github.io/hts-specs/SAMtags.pdf for format details.

    Args:
        seq (str): read-centric read sequence. For reference-anchored calls
            this should be the reverse complement sequence.
        poss (list): positions relative to seq
        probs (np.array): probabilties for modified bases
        mod_bases (str): modified base single letter codes
        can_base (str): canonical base

    Returns:
        MM string tag and ML array tag
    """


    mm_tag, ml_tag = "", array.array("B")

    # compute modified base positions relative to the running total of the
    # associated canonical base
    if(len(poss)==0):
        return f"{can_base}+{mod_base};",ml_tag
    can_base_mod_poss = (
        np.cumsum([1 if b == can_base else 0 for b in seq.replace("m","C")])[
            np.array(poss)
        ]
        - 1
    )
    mod_gaps = ",".join(
        map(str, np.diff(np.insert(can_base_mod_poss, 0, -1)) -1)
    )
    mm_tag += f"{can_base}+{mod_base},{mod_gaps};"
    # extract mod scores and scale to 0-255 range
    for _ in range(len(poss)):
        ml_tag.extend([255])

    return mm_tag,ml_tag

main(argparser().parse_args())