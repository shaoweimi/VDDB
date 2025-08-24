# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import pickle

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
import torchmetrics
from .util import unsqueeze_xdim
import distributed_util as dist_util
from evaluation import build_resnet50

from . import util
from .network import Image256Net,Image64Net
from .diffusion import Diffusion

from ipdb import set_trace as debug
from torch.cuda.amp import autocast, GradScaler
def build_optimizer_sched(opt, net, log, flag):

    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    #if opt.load_OT:
    if opt.load_OT and opt.load_DE:
        if flag == 'OT':
            checkpoint = torch.load(opt.load_OT, map_location="cpu")
            if "optimizer" in checkpoint.keys():
                optimizer.load_state_dict(checkpoint["optimizer"])
                log.info(f"[Opt] Loaded optimizer ckpt {opt.load_OT}!")
            else:
                log.warning(f"[Opt] Ckpt {opt.load_OT} has no optimizer!")
            if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
                sched.load_state_dict(checkpoint["sched"])
                log.info(f"[Opt] Loaded lr sched ckpt {opt.load_OT}!")
            else:
                log.warning(f"[Opt] Ckpt {opt.load_OT} has no lr sched!")
        elif flag == 'DE':
            checkpoint = torch.load(opt.load_DE, map_location="cpu")
            if "optimizer" in checkpoint.keys():
                optimizer.load_state_dict(checkpoint["optimizer"])
                log.info(f"[Opt] Loaded optimizer ckpt {opt.load_DE}!")
            else:
                log.warning(f"[Opt] Ckpt {opt.load_DE} has no optimizer!")
            if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
                sched.load_state_dict(checkpoint["sched"])
                log.info(f"[Opt] Loaded lr sched ckpt {opt.load_DE}!")
            else:
                log.warning(f"[Opt] Ckpt {opt.load_DE} has no lr sched!")
    return optimizer, sched

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()

class Runner(object):
    def __init__(self, opt, log, save_opt=True):
        super(Runner,self).__init__()

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
        # self.net_OT = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
        
        tensor_interval = torch.tensor(opt.interval, dtype=torch.float32)
        self.net_OT = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
        self.net_DE = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
        
        #self.net_OT = Image64Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
        # self.net_DE = Image64Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
        
        self.ema_OT = ExponentialMovingAverage(self.net_OT.parameters(), decay=opt.ema)
        self.ema_DE = ExponentialMovingAverage(self.net_DE.parameters(), decay=opt.ema)

        #if opt.load_OT:
        if opt.load_OT and opt.load_DE:
            checkpoint_OT = torch.load(opt.load_OT, map_location="cpu")
            self.net_OT.load_state_dict(checkpoint_OT['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load_OT}!")
            self.ema_OT.load_state_dict(checkpoint_OT["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load_OT}!")

            checkpoint_DE = torch.load(opt.load_DE, map_location="cpu")
            self.net_DE.load_state_dict(checkpoint_DE['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load_DE}!")
            self.ema_DE.load_state_dict(checkpoint_DE["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load_DE}!")
            
        self.net_OT.to(opt.device)
        self.net_DE.to(opt.device)
        self.ema_OT.to(opt.device)
        self.ema_DE.to(opt.device)
        self.log = log

    def compute_label_OT(self, opt, step, x0, μt):
        tensor_interval = torch.tensor(opt.interval, dtype=torch.float32)
        interval = unsqueeze_xdim(tensor_interval,xdim=x0.shape[1:]).to(opt.device)
        betas_step = self.diffusion.get_betas(step,xdim=x0.shape[1:])
        square_fwd = self.diffusion.get_square_fwd(step,xdim=x0.shape[1:])
        label_OT = betas_step * interval * (x0 - μt) / square_fwd
        
        return label_OT.detach()
        # return label_OT.detach()
    
    def compute_label_DE(self, xt, μt):
        
        label_DE = xt - μt
        
        return label_DE.detach()
    
    def compute_pred_x0(self, opt,step, xt, net_out, μt, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        # std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        # pred_x0 = xt - std_fwd * net_out
        # if clip_denoise: pred_x0.clamp_(-1., 1.)
        square_fwd= self.diffusion.get_square_fwd(step,xdim=xt.shape[1:])
        betas_step= self.diffusion.get_betas(step,xdim=xt.shape[1:])
        # pred_x0= μt + net_out * square_fwd / betas_step
        tensor_interval = torch.tensor(opt.interval, dtype=torch.float32)
        interval = unsqueeze_xdim(tensor_interval,xdim=μt.shape[1:]).to(opt.device)
        
        pred_x0 = μt + (square_fwd * net_out) / (interval * betas_step)
        
        return pred_x0

    def compute_pred_μt(self, step, xt, de_noise, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_sb = self.diffusion.get_std_sb(step, xdim=xt.shape[1:])
        μt = xt - std_sb * de_noise
        if clip_denoise: μt.clamp_(-1., 1.)
        
        
        return μt
    
    
    def sample_batch(self, opt, loader, corrupt_method):
        if opt.corrupt == "mixture":
            clean_img, corrupt_img, y = next(loader)
            mask = None
        elif "inpaint" in opt.corrupt:
            clean_img, y = next(loader)
            with torch.no_grad():
                corrupt_img, mask = corrupt_method(clean_img.to(opt.device))
        elif "color" in opt.corrupt:
            clean_img,corrupt_img,_ = next(loader)
            mask = None
        elif "normal" in opt.corrupt:
            clean_img,corrupt_img,_ = next(loader)
            mask = None
        else:
            clean_img, y = next(loader)
            with torch.no_grad():
                corrupt_img = corrupt_method(clean_img.to(opt.device))
            mask = None

        # os.makedirs(".debug", exist_ok=True)
        # tu.save_image((clean_img+1)/2, ".debug/clean.png", nrow=4)
        # tu.save_image((corrupt_img+1)/2, ".debug/corrupt.png", nrow=4)
        # debug()
        if "color" in opt.corrupt:
            y = None
        elif "normal" in opt.corrupt:
            y = None
        else:
            y  = y.detach().to(opt.device)
        x0 = clean_img.detach().to(opt.device)
        x1 = corrupt_img.detach().to(opt.device)
        if mask is not None:
            mask = mask.detach().to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)
        cond = x1.detach() if opt.cond_x1 else None

        if opt.add_x1_noise: # only for decolor
            x1 = x1 + torch.randn_like(x1)

        assert x0.shape == x1.shape

        return x0, x1, mask, y, cond

    def train(self, opt, train_dataset, val_dataset, corrupt_method):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net_OT = DDP(self.net_OT, device_ids=[opt.device])
        net_DE = DDP(self.net_DE, device_ids=[opt.device])
        
        ema_OT = self.ema_OT
        ema_DE = self.ema_DE
        
        
        optimizer_OT, sched_OT = build_optimizer_sched(opt, net_OT, log,'OT')
        optimizer_DE, sched_DE = build_optimizer_sched(opt, net_DE, log,'DE')
        
        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader   = util.setup_loader(val_dataset,   opt.microbatch)

        net_OT.train()
        net_DE.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        
        for it in range(opt.globals_it,opt.num_itr):
            optimizer_OT.zero_grad()
            #optimizer_DE.zero_grad()

            for _ in range(n_inner_loop):
                x0, x1, mask, y, cond = self.sample_batch(opt, train_loader, corrupt_method)

                step = torch.randint(0, opt.interval, (x0.shape[0],))

                μt, xt, std_sb= self.diffusion.q_sample(step, x0, x1)
                
                # OT = beta * (x0 - μt)/sigmat, sigmat*OT/beta = (x0 - μt)
                label_OT = self.compute_label_OT(opt, step, x0, μt)
                # xt - μt
                #label_DE = self.compute_label_DE(xt,μt)
                
                pred_OT = net_OT(xt, step, cond=cond)
                
                pred_DE = net_DE(xt, step, cond=cond)
                
                assert xt.shape == label_OT.shape ==label_DE.shape== pred_OT.shape == pred_DE.shape
                if mask is not None:
                    pred_OT = mask * pred_OT
                    pred_DE = mask * pred_DE
                    label_OT = mask * label_OT
                    label_DE = mask * label_DE
                    
                    
                loss_OT = F.mse_loss(pred_OT,label_OT)
                #loss_DE = F.mse_loss(std_sb*pred_DE,label_DE)
                loss_OT.backward()
                #loss_DE.backward()
                
            optimizer_OT.step()
            optimizer_DE.step()
            ema_OT.update()
            ema_DE.update()
            if sched_OT is not None: sched_OT.step()
            #if sched_DE is not None: sched_DE.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr_OT:{} | loss_OT:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer_OT.param_groups[0]['lr']),
                "{:+.4f}".format(loss_OT.item()),
                "{:.2e}".format(optimizer_DE.param_groups[0]['lr']),
                "{:+.4f}".format(loss_DE.item()),
            ))
            if (it+1) % 10 == 0:
                self.writer.add_scalar(it, 'loss_OT', loss_OT.detach())
                self.writer.add_scalar(it, 'loss_DE', loss_DE.detach())
                
            if (it+1) % 5000 == 0:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net_OT.state_dict(),
                        "ema": ema_OT.state_dict(),
                        "optimizer": optimizer_OT.state_dict(),
                        "sched": sched_OT.state_dict() if sched_OT is not None else sched_OT,
                    }, opt.ckpt_path / f"{opt.task}_OT_{it}.pt")
                     torch.save({
                          "net": self.net_DE.state_dict(),
                          "ema": ema_DE.state_dict(),
                          "optimizer": optimizer_DE.state_dict(),
                          "sched": sched_DE.state_dict() if sched_DE is not None else sched_DE,
                     }, opt.ckpt_path / f"{opt.task}_DE_{it}.pt")
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                    
                if opt.distributed:
                    torch.distributed.barrier()

             if (it+1) == 500 or (it+1) % 5000 == 0: # 0, 0.5k, 3k, 6k 9k
                 net_OT.eval()
                 net_DE.eval()
                 self.evaluation(opt, it, val_loader, corrupt_method)
                 net_OT.train()
                 net_DE.train()
        self.writer.close()

    @torch.no_grad()
    def ddpm_sampling(self, opt, x1, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True):

        nfe = nfe or opt.interval-1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe+1)

        # create log steps
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0
        self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)
        if mask is not None:
            mask = mask.to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.ema_OT.average_parameters():
            with self.ema_DE.average_parameters():
                self.net_OT.eval()
                self.net_DE.eval()
                
                def pred_xt_noise(xt,step):
                    step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                    de_noise = self.net_DE(xt,step)
                    return self.compute_pred_μt(step,xt,de_noise,clip_denoise=clip_denoise)
                    
                def pred_x0_fn(xt, step, μt):
                    step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                    net_OT_out = self.net_OT(xt, step, cond=cond)
                    return self.compute_pred_x0(opt,step, xt, net_OT_out, μt,clip_denoise=clip_denoise)

                xs, pred_x0 = self.diffusion.ddpm_sampling(
                    steps, pred_x0_fn,pred_xt_noise, x1, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose,
                )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return xs, pred_x0

    @torch.no_grad()
    def evaluation(self, opt, it, val_loader, corrupt_method):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        img_clean, img_corrupt, mask, y, cond = self.sample_batch(opt, val_loader, corrupt_method)

        x1 = img_corrupt.to(opt.device)

        xs, pred_x0s = self.ddpm_sampling(
            opt, x1, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, verbose=opt.global_rank==0
        )

        log.info("Collecting tensors ...")
        img_clean   = all_cat_cpu(opt, log, img_clean)
        img_corrupt = all_cat_cpu(opt, log, img_corrupt)
        if "color" not in opt.corrupt:
            y       = all_cat_cpu(opt, log, y)
        xs          = all_cat_cpu(opt, log, xs)
        pred_x0s    = all_cat_cpu(opt, log, pred_x0s)

        batch, len_t, *xdim = xs.shape
        assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
        assert xs.shape == pred_x0s.shape
        if "color" not in opt.corrupt:
            assert y.shape == (batch,)
        
        log.info(f"Generated recon trajectories: size={xs.shape}")

        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid((img+1)/2, nrow=nrow)) # [1,1] -> [0,1]



        log.info("Logging images ...")
        img_recon = xs[:, 0, ...]
        loss = F.mse_loss(img_recon,img_clean)
        self.writer.add_scalar(it, "val_loss", loss.detach())
        
        
        log_image("image/clean",   img_clean)
        log_image("image/corrupt", img_corrupt)
        log_image("image/recon",   img_recon)
        log_image("debug/pred_clean_traj", pred_x0s.reshape(-1, *xdim), nrow=len_t)
        log_image("debug/recon_traj",      xs.reshape(-1, *xdim),      nrow=len_t)



        log.info(f"========== Evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()
