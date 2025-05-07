#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test if KL autoencoder works with steganography
@author: Tu Bui @University of Surrey
"""
import os, sys, torch 
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from tools.helpers import welcome_message


def trainer_settings(config, output_dir):
    out = {}
    ckptdir = os.path.join(output_dir, 'checkpoints')
    cfgdir = os.path.join(output_dir, 'configs')
    if os.path.exists(os.path.join(ckptdir, 'last.ckpt')):
        resumedir = output_dir
        out['resume_from_checkpoint'] = os.path.join(ckptdir, 'last.ckpt')
    else:
        resumedir = ''

    pl_config = config.get("lightning", OmegaConf.create())
    # callbacks
    callbacks = {
        'generic': dict(target='cldm.logger.SetupCallback', 
        params={'resume': resumedir, 'now': '', 'logdir': output_dir, 'ckptdir': ckptdir, 'cfgdir': cfgdir, 'config': config, 'lightning_config': pl_config}),

        'cuda': dict(target='cldm.logger.CUDACallback', params={}),

        'ckpt': dict(target='pytorch_lightning.callbacks.ModelCheckpoint',
        params={'dirpath': ckptdir, 'filename': '{epoch:06}', 'verbose': True, 'save_top_k': -1, 'save_last': True}),
     
    }
    if 'checkpoint' in pl_config.callbacks:
        callbacks['ckpt'] = OmegaConf.merge(callbacks['ckpt'], pl_config.callbacks.checkpoint)

    if 'progress_bar' in pl_config.callbacks:
        callbacks['probar'] = pl_config.callbacks.progress_bar

    if 'image_logger' in pl_config.callbacks:
        callbacks['img_log'] = pl_config.callbacks.image_logger

    callbacks = [instantiate_from_config(c) for k, c in callbacks.items()]
    out['callbacks'] = callbacks

    # logger
    logger = dict(target='pytorch_lightning.loggers.TestTubeLogger', params={'name': 'testtube', 'save_dir': output_dir})
    logger = instantiate_from_config(logger)
    out['logger'] = logger

    return out

def get_learningrate(pl_config, args):
    base_lr = pl_config.trainer.base_learning_rate
    lr = base_lr * args.gpus * args.batch_size
    if 'accumulate_grad_batches' in pl_config.trainer:
        lr *= pl_config.trainer.accumulate_grad_batches
        grad_batches = pl_config.trainer.accumulate_grad_batches
    else:
        grad_batches = 1
    print(f'Learning rate set to: {lr:.2e} = {base_lr:.2e} (base lr) x {args.batch_size} (batch size) x {grad_batches} (accumulate_grad_batches) x {args.gpus} (gpus)')
    return lr


def get_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='models/AE.yaml')
    parser.add_argument('-o', '--output', type=str, default='/mnt/fast/nobackup/scratch4weeks/tb0035/projects/diffsteg/controlnet/AE')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_image_weight_ratio', type=float, default=2., help='max weight of image loss after ramping')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size, 8 for 1 A100 80GB GPU')
    return parser.parse_args()

def app(args):
    output = args.output
    config = OmegaConf.load(args.config)
    config.model.params.loss_config.params.max_image_weight_ratio = args.max_image_weight_ratio
    
    # data
    data_config = config.get("data", OmegaConf.create())  # config.pop()
    data_config.params.batch_size = args.batch_size

    # resolution = 256
    data = instantiate_from_config(data_config)
    # tform = transforms.Resize((resolution,resolution))
    data.prepare_data()
    data.setup()
    # for k in data.datasets:
    #     data.datasets[k].set_transform(tform)
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # trainer
    trainer_kwargs = dict(gpus=args.gpus, precision=32)
    trainer_kwargs.update(trainer_settings(config, output))
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.logdir = output

    # model
    model = instantiate_from_config(config.model).cpu()
    model.learning_rate = get_learningrate(config.lightning, args)

    # Train!
    trainer.fit(model, data)

if __name__ == '__main__':
    print(welcome_message())
    args = get_parser()
    app(args)