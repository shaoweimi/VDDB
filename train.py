from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import random
import argparse

import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.multiprocessing import Process
from dataset import diode
from logger import Logger
from distributed_util import init_processes
from corruption import build_corruption
from dataset import imagenet
from i2sb import Runner, download_ckpt
#from dataset import edges2handbags
import colored_traceback.always
from ipdb import set_trace as debug
import re
import torch.multiprocessing as mp
RESULT_DIR = Path("results")

def load_max_checkpoint(folder_path, model_prefixes):
    pattern = re.compile(r'_(\d+)\.pt$') 

    steps_dict = {}
    for filename in os.listdir(folder_path):
        match = pattern.search(filename)
        if match:
            num = int(match.group(1))
            for prefix in model_prefixes:
                if filename.startswith(prefix):
                    steps_dict.setdefault(num, set()).add(prefix)

    valid_steps = [step for step, prefixes in steps_dict.items() if set(prefixes) == set(model_prefixes)]

    if not valid_steps:
        print("Not find .pt")
        return None, 0

    max_num = max(valid_steps)
    max_file_paths = {prefix: os.path.join(folder_path, f"{prefix}_{max_num}.pt")
                      for prefix in model_prefixes}

    print(f"max step={max_num}, file_path: {max_file_paths}")

    start_step = max_num + 1
    return max_file_paths, start_step
def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def create_training_options():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--name",           type=str,   default=None,        help="experiment ID")
    parser.add_argument("--ckpt",           type=str,   default=None,        help="resumed checkpoint name")
    parser.add_argument("--gpu",            type=int,   default=None,        help="set only if you wish to run on a particular device")
    parser.add_argument("--n-gpu-per-node", type=int,   default=2,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,   default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,   default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,   default=1,           help="The number of nodes in multi node env")
    # parser.add_argument("--amp",            action="store_true")

    # --------------- SB model ---------------
    parser.add_argument("--image-size",     type=int,   default=256)
    parser.add_argument("--corrupt",        type=str,   default=None,        help="restoration task")
    parser.add_argument("--t0",             type=float, default=1e-4,        help="sigma start time in network parametrization")
    parser.add_argument("--T",              type=float, default=1.,          help="sigma end time in network parametrization")
    parser.add_argument("--interval",       type=int,   default=1000,        help="number of interval")
    parser.add_argument("--beta-max",       type=float, default=0.3,         help="max diffusion for the diffusion model")
    # parser.add_argument("--beta-min",       type=float, default=0.1)
    parser.add_argument("--ot-ode",         action="store_true",             help="use OT-ODE model")
    parser.add_argument("--clip-denoise",   action="store_true",             help="clamp predicted image to [-1,1] at each")

    # optional configs for conditional network
    parser.add_argument("--cond-x1",        action="store_true",             help="conditional the network on degraded images")
    parser.add_argument("--add-x1-noise",   action="store_true",             help="add noise to conditional network")

    # --------------- optimizer and loss ---------------
    parser.add_argument("--batch-size",     type=int,   default=64)
    parser.add_argument("--microbatch",     type=int,   default=1,           help="accumulate gradient over microbatch until full batch-size")
    parser.add_argument("--num-itr",        type=int,   default=150000,       help="training iteration")
    parser.add_argument("--lr",             type=float, default=5e-5,        help="learning rate")
    parser.add_argument("--lr-gamma",       type=float, default=0.99,        help="learning rate decay ratio")
    parser.add_argument("--lr-step",        type=int,   default=1000,        help="learning rate decay step size")
    parser.add_argument("--l2-norm",        type=float, default=0.0)
    parser.add_argument("--ema",            type=float, default=0.99)

    # --------------- path and logging ---------------
    parser.add_argument("--dataset-dir",    type=Path,  default="/dev/shm/datasets",  help="path to LMDB dataset")
    parser.add_argument("--task",           type=str,   default="diode", help="restoration task")
    parser.add_argument("--log-dir",        type=Path,  default=".log",      help="path to log std outputs and writer data")
    parser.add_argument("--log-writer",     type=str,   default="wandb",        help="log writer: can be tensorbard, wandb, or None")
    parser.add_argument("--wandb-api-key",  type=str,   default="c87a7bbbdcd839fa32ed7848bae21548b75389c0",        help="unique API key of your W&B account; see https://wandb.ai/authorize")
    parser.add_argument("--wandb-user",     type=str,   default=None,        help="user name of your W&B account")

    opt = parser.parse_args()

    # ========= auto setup =========
    opt.device='cuda' if opt.gpu is None else f'cuda:{opt.gpu}'
    if opt.name is None:
        opt.name = opt.corrupt
    opt.distributed = opt.n_gpu_per_node > 1
    opt.use_fp16 = False # disable fp16 for training

    # log ngc meta data
    if "NGC_JOB_ID" in os.environ.keys():
        opt.ngc_job_id = os.environ["NGC_JOB_ID"]

    # ========= path handle =========
    os.makedirs(opt.log_dir, exist_ok=True)
    opt.ckpt_path = RESULT_DIR / opt.name
    os.makedirs(opt.ckpt_path, exist_ok=True)

    if opt.ckpt is not None:
        ckpt_file = RESULT_DIR / opt.ckpt 
        model_prefixes = ["diode_OT_", "diode_DE_"]  
        checkpoints, start_step = load_max_checkpoint(ckpt_file, model_prefixes)

        assert ckpt_file.exists()
        if checkpoints is not None:
            opt.load_OT = checkpoints["diode_OT_"]
            opt.load_DE = checkpoints["diode_DE_"]
            opt.globals_it = start_step
        else:
            opt.load_OT = None
            opt.load_DE = None
            opt.globals_it= 0
    else:
        opt.load_OT = None
        opt.load_DE = None
        opt.globals_it = 0
    # opt.globals_it = 0
    # opt.load_OT = None
    # opt.load_DE = None
    # ========= auto assert =========
    assert opt.batch_size % opt.microbatch == 0, f"{opt.batch_size=} is not dividable by {opt.microbatch}!"
    return opt

def main(opt):
    log = Logger(opt.global_rank, opt.log_dir)
    log.info("=======================================================")
    log.info("         Image-to-Image Schrodinger Bridge")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")

    # set seed: make sure each gpu has differnet seed!
    if opt.seed is not None:
        set_seed(opt.seed + opt.global_rank)

    # build imagenet dataset
    # train_dataset = imagenet.build_lmdb_dataset(opt, log, train=True)
    # val_dataset   = imagenet.build_lmdb_dataset(opt, log, train=False)
    # note: images should be normalized to [-1,1] for corruption methods to work properly
    # root = opt.dataset_dir / "edges2handbags"
    # train_dataset = edges2handbags.EdgesDataset(dataroot=root, train=True, img_size=opt.image_size, random_crop=True, random_flip=True)
    # val_dataset = edges2handbags.EdgesDataset(dataroot=root, train=False, img_size=opt.image_size, random_crop=False, random_flip=False)
    
    
    root = opt.dataset_dir / "DIODE-256"
    train_dataset = diode.DIODE(dataroot=root, train=True, img_size=opt.image_size, random_crop=False, random_flip=True)
    val_dataset = diode.DIODE(dataroot=root, train=True, img_size=opt.image_size, random_crop=False, random_flip=True)
    if opt.corrupt == "mixture":
        import corruption.mixture as mix
        train_dataset = mix.MixtureCorruptDatasetTrain(opt, train_dataset)
        val_dataset = mix.MixtureCorruptDatasetVal(opt, val_dataset)

    # build corruption method
    corrupt_method = build_corruption(opt, log)

    run = Runner(opt, log)
    run.train(opt, train_dataset, val_dataset, corrupt_method)
    log.info("Finish!")

if __name__ == '__main__':
    
    mp.set_start_method('spawn')
    opt = create_training_options()

    assert opt.corrupt is not None

    # one-time download: ADM checkpoint
    download_ckpt("data/")

    if opt.distributed:
        size = opt.n_gpu_per_node

        processes = []
        for rank in range(size):
            opt = copy.deepcopy(opt)
            opt.local_rank = rank
            global_rank = rank + opt.node_rank * opt.n_gpu_per_node
            global_size = opt.num_proc_node * opt.n_gpu_per_node
            opt.global_rank = global_rank
            opt.global_size = global_size
            print('Node rank %d, local proc %d, global proc %d, global_size %d' % (opt.node_rank, rank, global_rank, global_size))
            p = Process(target=init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        init_processes(0, opt.n_gpu_per_node, main, opt)
