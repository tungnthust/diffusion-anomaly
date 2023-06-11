"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os
import sys
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.litsloader import LiTSDataset

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
# from visdom import Visdom
import numpy as np
# viz = Visdom(port=8850)
# loss_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='classification loss'))
# val_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='validation loss'))
# acc_window= viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='acc', title='accuracy'))

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.script_util import (
    add_dict_to_argparser,

)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.losses import DiceCELoss

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    print(args)
    model = SwinUNETR(img_size=(args.image_size, args.image_size), 
                      in_channels=1, 
                      out_channels=1, 
                      use_checkpoint=True, 
                      spatial_dims=2)
    model.to(dist_util.dev())
    
    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.use_fp16, initial_lg_loss_scale=16.0
    )

    
    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )
        
    logger.log("creating data loader...")

    dice_loss = DiceCELoss(to_onehot_y=False, sigmoid=True)

    print("Training on LiTS dataset")

    ds = LiTSDataset(mode="train", fold=args.fold, test_flag=False)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)

    try:
        val_ds = LiTSDataset(mode="test", fold=args.fold, test_flag=False)
        val_datal = th.utils.data.DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=True)
        val_data = iter(val_datal)
    except:
        val_data = None

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training classifier model...")

    def validation_log(val_data_load):
        data_loader = iter(val_data_load)
        dice_scores = []
        losses = []
        data_size = 0
        for data in data_loader:
            batch, _, _, liver_masks, _ = data
            data_size += batch.shape[0]
            batch = batch.to(dist_util.dev())
            liver_masks = liver_masks.to(dist_util.dev())
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
            for i, (sub_batch, sub_liver_masks) in enumerate(
                split_microbatches(args.microbatch, batch, liver_masks)
            ):
            
                logits = model(sub_batch)
            
                loss = dice_loss(logits, sub_liver_masks)
                loss = loss.mean()

                losses.append(loss.mean().item())
                dice_scores.append(dice_score(logits, sub_liver_masks))
                
        print(f"Validation dataset size: {data_size}")

        return np.mean(losses), np.mean(dice_scores)
    
    def forward_backward_log(data_load, data_loader, prefix="train"):
        try:
            batch, _, _, liver_masks, _ = next(data_loader)
        except:
            data_loader = iter(data_load)
            batch, _, _, liver_masks, _ = next(data_loader)

        # print('labels', labels)
        batch = batch.to(dist_util.dev())
        liver_masks= liver_masks.to(dist_util.dev())

        for i, (sub_batch, sub_liver_masks) in enumerate(
            split_microbatches(args.microbatch, batch, liver_masks)
        ):
          
            logits = model(sub_batch)
         
            # loss = F.cross_entropy(logits, sub_labels, reduction="none")
            loss = dice_loss(logits, sub_liver_masks)

            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_dice"] = dice_score(logits, sub_liver_masks)

            # losses[f"{prefix}_acc@2"] = compute_top_k(
            #     logits, sub_labels, k=2, reduction="none"
            # )
            # print('acc', losses[f"{prefix}_acc@1"])
            for key, values in losses.items():
                logger.logkv_mean(key, values.mean().item())
            loss = loss.mean()
#             if prefix=="train":
#                 pass
# #                 viz.line(X=th.ones((1, 1)).cpu() * step, Y=th.Tensor([loss]).unsqueeze(0).cpu(),
# #                      win=loss_window, name='loss_cls',
# #                      update='append')

#             else:

#                 output_idx = logits[0].argmax()
#                 print('outputidx', output_idx)
#                 output_max = logits[0, output_idx]
#                 print('outmax', output_max, output_max.shape)
#                 output_max.backward()
#                 saliency, _ = th.max(sub_batch.grad.data.abs(), dim=1)
#                 print('saliency', saliency.shape)
# #                 viz.heatmap(visualize(saliency[0, ...]))
# #                 viz.image(visualize(sub_batch[0, 0,...]))
# #                 viz.image(visualize(sub_batch[0, 1, ...]))
#                 th.cuda.empty_cache()


            if loss.requires_grad and prefix=="train":
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

        return losses

    correct=0; total=0
    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        # print('step', step + resume_step)
        
        losses = forward_backward_log(datal, data)

        
        mp_trainer.optimize(opt)
        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_datal, val_data, prefix="val")
                    val_loss, val_dice = validation_log(val_datal)
                    print(f"Validation loss: {val_loss} - Validation DICE: {val_dice}")
                    model.train()

        if not step % args.log_interval:
            print('step', step + resume_step)
            logger.dumpkvs()
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"liver_seg/model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"liver_seg/opt{step:06d}.pt"))

def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)

def dice_score(logits, targs):
    pred = th.sigmoid(logits)
    pred = (pred > 0.5).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def create_argparser():
    defaults = dict(
        data_dir="",
        image_size=256,
        val_data_dir="",
        iterations=150000,
        lr=1e-4,
        weight_decay=0.0,
        anneal_lr=False,
        use_fp16=False,
        batch_size=4,
        microbatch=-1,
        resume_checkpoint="",
        log_interval=10,
        eval_interval=1000,
        save_interval=10000,
        fold=1
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
