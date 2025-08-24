import os
import copy
import argparse
import random
from pathlib import Path
from easydict import EasyDict as edict

import numpy as np
#from dataset import edges2handbags
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
from dataset import diode
from logger import Logger
import distributed_util as dist_util
from i2sb import Runner, download_ckpt
from corruption import build_corruption
from dataset import imagenet
from i2sb import ckpt_util
import multiprocessing as mp
import colored_traceback.always
from ipdb import set_trace as debug

RESULT_DIR = Path("results")

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def build_subset_per_gpu(opt, dataset, log):
    n_data = len(dataset)
    n_gpu  = opt.global_size
    n_dump = (n_data % n_gpu > 0) * (n_gpu - n_data % n_gpu)

    # create index for each gpu
    total_idx = np.concatenate([np.arange(n_data), np.zeros(n_dump)]).astype(int)
    idx_per_gpu = total_idx.reshape(-1, n_gpu)[:, opt.global_rank]
    log.info(f"[Dataset] Add {n_dump} data to the end to be devided by {n_gpu=}. Total length={len(total_idx)}!")

    # build subset
    indices = idx_per_gpu.tolist()
    subset = Subset(dataset, indices)
    log.info(f"[Dataset] Built subset for gpu={opt.global_rank}! Now size={len(subset)}!")
    return subset

def collect_all_subset(sample, log):
    batch, *xdim = sample.shape
    gathered_samples = dist_util.all_gather(sample, log)
    gathered_samples = [sample.cpu() for sample in gathered_samples]
    # [batch, n_gpu, *xdim] --> [batch*n_gpu, *xdim]
    return torch.stack(gathered_samples, dim=1).reshape(-1, *xdim)

def build_partition(opt, full_dataset, log):
    n_samples = len(full_dataset)

    part_idx, n_part = [int(s) for s in opt.partition.split("_")]
    assert part_idx < n_part and part_idx >= 0
    assert n_samples % n_part == 0

    n_samples_per_part = n_samples // n_part
    start_idx = part_idx * n_samples_per_part
    end_idx = (part_idx+1) * n_samples_per_part

    indices = [i for i in range(start_idx, end_idx)]
    subset = Subset(full_dataset, indices)
    log.info(f"[Dataset] Built partition={opt.partition}, {start_idx=}, {end_idx=}! Now size={len(subset)}!")
    return subset

def build_val_dataset(opt, log, corrupt_type):
    if "sr4x" in corrupt_type:
        val_dataset = imagenet.build_lmdb_dataset(opt, log, train=False) # full 50k val
    elif "inpaint" in corrupt_type:
        mask = corrupt_type.split("-")[1]
        val_dataset = imagenet.InpaintingVal10kSubset(opt, log, mask) # subset 10k val + mask
    elif corrupt_type == "mixture":
        from corruption.mixture import MixtureCorruptDatasetVal
        val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log)
        val_dataset = MixtureCorruptDatasetVal(opt, val_dataset) # subset 10k val + mixture
    else:
        val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log) # subset 10k val

    # build partition
    if opt.partition is not None:
        val_dataset = build_partition(opt, val_dataset, log)
    return val_dataset

def get_recon_imgs_fn(opt, nfe):
    sample_dir = RESULT_DIR / opt.ckpt / "samples_nfe{}{}".format(
        nfe, "_clip" if opt.clip_denoise else ""
    )
    os.makedirs(sample_dir, exist_ok=True)

    recon_imgs_fn = sample_dir / "recon{}.pt".format(
        "" if opt.partition is None else f"_{opt.partition}"
    )
    return recon_imgs_fn

def compute_batch(ckpt_opt, corrupt_type, corrupt_method, out):
    if "inpaint" in corrupt_type:
        clean_img, y, mask = out
        corrupt_img = clean_img * (1. - mask) + mask
        x1          = clean_img * (1. - mask) + mask * torch.randn_like(clean_img)
    elif corrupt_type == "mixture":
        clean_img, corrupt_img, y = out
        mask = None
    elif corrupt_type == "color":
        clean_img, corrupt_img, _ = out
        mask = None
        x1 = corrupt_img.to(ckpt_opt.device)
        y=None
    elif corrupt_type == "normal":
        clean_img,corrupt_img,_ = out
        mask = None
        x1 = corrupt_img.to(ckpt_opt.device)
        y=None
    else:
        clean_img, y = out
        mask = None
        corrupt_img = corrupt_method(clean_img.to(opt.device))
        x1 = corrupt_img.to(opt.device)

    cond = x1.detach() if ckpt_opt.cond_x1 else None
    if ckpt_opt.add_x1_noise: # only for decolor
        x1 = x1 + torch.randn_like(x1)

    return clean_img.to(ckpt_opt.device), x1, mask, cond, y

@torch.no_grad()
def main(opt):
    log = Logger(opt.global_rank, ".log")

    # get (default) ckpt option
    ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / opt.ckpt)
    corrupt_type = ckpt_opt.corrupt
    nfe = opt.nfe or ckpt_opt.interval-1

    # build corruption method
    corrupt_method = build_corruption(opt, log, corrupt_type=corrupt_type)
    if opt.ckpt is not None:
        ckpt_file = RESULT_DIR / opt.ckpt 
        prefixes = ['diode_OT_', 'diode_DE_']
        # checkpoints = load_max_checkpoints(ckpt_file, prefixes)

        # ot_ckpt_path, ot_step = checkpoints['edges2handbags_OT_']
        # de_ckpt_path, de_step = checkpoints['edges2handbags_DE_']
        assert ckpt_file.exists()
        ckpt_opt.load_OT = RESULT_DIR / opt.ckpt / 'diode_OT_149999.pt'
        ckpt_opt.load_DE = RESULT_DIR / opt.ckpt / 'diode_DE_149999.pt'
        # opt.globals_it = ot_step
    else:
        opt.load_OT = None
        opt.load_DE = None
    # build imagenet val dataset
    # root = opt.dataset_dir / "edges2handbags"
    # #train_dataset = edges2handbags.EdgesDataset(dataroot=root, train=True, img_size=opt.image_size, random_crop=True, random_flip=True)
    # val_dataset = edges2handbags.EdgesDataset(dataroot=root, train=True, img_size=opt.image_size, random_crop=False, random_flip=False)
    root = opt.dataset_dir / "DIODE-256"
    val_dataset = diode.DIODE(dataroot=root, train=True, img_size=opt.image_size, random_crop=False, random_flip=False)
    
    #val_dataset = build_val_dataset(opt, log, corrupt_type)
    n_samples = len(val_dataset)

    # build dataset per gpu and loader
    subset_dataset = build_subset_per_gpu(opt, val_dataset, log)
    val_loader = DataLoader(subset_dataset,
        batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,
    )

    # build runner
    runner = Runner(ckpt_opt, log, save_opt=False)

    # handle use_fp16 for ema
    if opt.use_fp16:
        runner.ema.copy_to() # copy weight from ema to net
        runner.net.diffusion_model.convert_to_fp16()
        runner.ema = ExponentialMovingAverage(runner.net.parameters(), decay=0.99) # re-init ema with fp16 weight

    # create save folder
    recon_imgs_fn = get_recon_imgs_fn(opt, nfe)
    log.info(f"Recon images will be saved to {recon_imgs_fn}!")
    recon_imgs = []
    clean_imgs = []
    ys = []
    num = 0

    for loader_itr, out in enumerate(val_loader):

        clean_img, x1, mask, cond, y = compute_batch(ckpt_opt, corrupt_type, corrupt_method, out)

        xs, _ = runner.ddpm_sampling(
            ckpt_opt, x1, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, nfe=nfe, verbose=opt.n_gpu_per_node==1
        )
        recon_img = xs[:, 0, ...].to(opt.device)

        assert recon_img.shape == clean_img.shape

        if  opt.global_rank == 0:  # debug
            os.makedirs(".debug", exist_ok=True)
            tu.save_image((clean_img+1)/2, f".debug/clean_img_{loader_itr}.png")
            tu.save_image((recon_img+1)/2, f".debug/recon_{loader_itr}.png")
            tu.save_image((x1+1)/2, f".debug/x1_{loader_itr}.png")
            log.info("Saved debug images!")

        gathered_recon_img = collect_all_subset(recon_img, log)
        gathered_recon_img_cpu = ((gathered_recon_img + 1 )*127.5).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        
        gathered_clean_img = collect_all_subset(clean_img, log)
        gathered_clean_img_cpu = ((gathered_clean_img + 1 )*127.5).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        
        
        recon_imgs.append(gathered_recon_img_cpu)
        clean_imgs.append(gathered_clean_img_cpu)
        
        num += len(gathered_recon_img)
        log.info(f"Collected {num} recon images!")
        dist.barrier()

    del runner

    # arr = torch.cat(recon_imgs, axis=0)[:n_samples]  # 不再裁剪到 total_required
    # arr2 = torch.cat(clean_img, axis=0)[:n_samples]
    
    arr = np.concatenate(recon_imgs, axis=0)[:n_samples]
    arr2 = np.concatenate(clean_imgs, axis=0)[:n_samples]
    # label_arr = torch.cat(ys, axis=0)
    if opt.global_rank == 0:
        # torch.save({"arr": arr}, recon_imgs_fn)
        # log.info(f"Save at {recon_imgs_fn}")
        # torch.save({"arr": arr2}, clean_img)
        # log.info(f"Save at {clean_img}")
        np.savez("clean_img_fid", arr_0=arr2)
        np.savez("recon_img_fid", arr_0=arr)
    dist.barrier()

    log.info(f"Sampling complete! Collect recon_imgs={arr.shape}")
    # recon_imgs = []
    # ys = []
    # num = 0
    # total_required = 10000
    # for loader_itr, out in enumerate(val_loader):

    #     corrupt_img, x1, mask, cond, y = compute_batch(ckpt_opt, corrupt_type, corrupt_method, out)

    #     xs, _ = runner.ddpm_sampling(
    #         ckpt_opt, x1, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, nfe=nfe, verbose=opt.n_gpu_per_node==1
    #     )
    #     recon_img = xs[:, 0, ...].to(opt.device)

    #     assert recon_img.shape == corrupt_img.shape

    #     if loader_itr == 0 and opt.global_rank == 0: # debug
    #         os.makedirs(".debug", exist_ok=True)
    #         tu.save_image((corrupt_img+1)/2, ".debug/corrupt.png")
    #         tu.save_image((recon_img+1)/2, ".debug/recon.png")
    #         log.info("Saved debug images!")

    #     # [-1,1]
    #     gathered_recon_img = collect_all_subset(recon_img, log)
    #     recon_imgs.append(gathered_recon_img)

    #     # y = y.to(opt.device)
    #     # gathered_y = collect_all_subset(y, log)
    #     # ys.append(gathered_y)

    #     num += len(gathered_recon_img)
    #     log.info(f"Collected {num} recon images!")
    #     dist.barrier()
    #     if num >= total_required:
    #         break
    # del runner

    # arr = torch.cat(recon_imgs, axis=0)[:total_required]
    # # label_arr = torch.cat(ys, axis=0)[:n_samples]

    # if opt.global_rank == 0:
    #     torch.save({"arr": arr}, recon_imgs_fn)
    #     log.info(f"Save at {recon_imgs_fn}")
    # dist.barrier()

    # log.info(f"Sampling complete! Collect recon_imgs={arr.shape}")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,  default=0)
    parser.add_argument("--n-gpu-per-node", type=int,  default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,  default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,  default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,  default=1,           help="The number of nodes in multi node env")
    parser.add_argument("--interval",       type=int,   default=1000,        help="number of interval")
    parser.add_argument("--beta-max",       type=float, default=1.0,         help="max diffusion for the diffusion model")
    # data
    parser.add_argument("--image-size",     type=int,  default=256)
    parser.add_argument("--dataset-dir",    type=Path, default="assets/datasets",  help="path to LMDB dataset")
    parser.add_argument("--partition",      type=str,  default=None,        help="e.g., '0_4' means the first 25% of the dataset")

    # sample
    parser.add_argument("--batch-size",     type=int,  default=4)
    parser.add_argument("--ckpt",           type=str,  default=None,        help="the checkpoint name from which we wish to sample")
    parser.add_argument("--nfe",            type=int,  default=None,        help="sampling steps")
    parser.add_argument("--clip-denoise",   action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--use-fp16",       action="store_true",            help="use fp16 network weight for faster sampling")

    arg = parser.parse_args()

    opt = edict(
        distributed=(arg.n_gpu_per_node > 1),
        device="cuda",
    )
    opt.update(vars(arg))

    # one-time download: ADM checkpoint
    #download_ckpt("data/")

    set_seed(opt.seed)

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
            p = Process(target=dist_util.init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        dist_util.init_processes(0, opt.n_gpu_per_node, main, opt)
