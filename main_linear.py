# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os

import torch
import torch.nn as nn
import json
from pytorch_lightning import Trainer
from pathlib import Path
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torchvision.models import resnet18, resnet50
from solo.methods import METHODS
from solo.args.setup import parse_args_linear
from solo.methods.base import BaseMethod
from solo.utils.backbones import (
    swin_base,
    swin_large,
    swin_small,
    swin_tiny,
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
)

try:
    from solo.methods.dali import ClassificationABC
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True
import types

from solo.methods.linear import LinearModel
from solo.utils.checkpointer import Checkpointer
from solo.utils.classification_dataloader import prepare_data


def main():
    args = parse_args_linear()

    ckpt_path = args.pretrained_feature_extractor
    print(ckpt_path)
    if not os.path.isfile(ckpt_path):
        print("No such file or directory")
        exit()

    assert (
        args.pretrained_feature_extractor.endswith(".ckpt")
        or args.pretrained_feature_extractor.endswith(".pth")
        or args.pretrained_feature_extractor.endswith(".pt")
    )
    print("==================================")
    print("checkpoint at: {}".format(ckpt_path))
   
    
    #checkpoint directory
    ckpt_dir = "/".join(ckpt_path.split("/")[:-1])
    print("Checkpoint directory at: {}".format(ckpt_dir))
    print("===================================")
    
    args_path = os.path.join(ckpt_dir, "args.json")


    # load args more model specific arguments
    if os.path.isfile(args_path):
        with open(args_path) as f:
            method_args = json.load(f)
    
    print(method_args['method'])
    

    
    state = torch.load(ckpt_path)["state_dict"]

    MethodClass = METHODS[method_args['method']]  

    
    model = MethodClass(**method_args)

    
    model.load_state_dict(state, strict=True)
    

    print(f"loaded:{ckpt_path}")
    
    if args.transfer:
        #update class information for the dataset on which transfer learning is being applied on
        if args.dataset=="cifar10" or  args.dataset=="stl10":
            method_args['num_classes'] = 10
        if args.dataset=="cifar100" or args.dataset=="imagenet100":
            method_args['num_classes'] = 100

    print("num_classes==",method_args['num_classes'])
    

    model = LinearModel(model, **args.__dict__)
    if args.transfer:
        print("checking for path for dataset")
        if args.dataset =="stl10" or args.dataset =="imagenet100":
           args.data_dir = "/home/tarun/Documents/PhD/modified_ssl_dgc/bash_files/pretrain/datasets"  



    train_loader, val_loader = prepare_data(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    callbacks = []

    # lr logging
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)
    
    #saving linear checkpoint 
    if args.save_checkpoint:
        # save checkpoint on last epoch only
        if args.transfer:
           #make dataset folder as checkpoint dir for clarity 
           args.checkpoint_dir = args.dataset

        ckpt_ = os.path.join(args.checkpoint_dir, "linear",  args.name, "run_"+str(args.run))
        if 'den_target' in method_args:
            #make change here for den_target
            ckpt_ = os.path.join(args.checkpoint_dir, "linear", str(method_args['den_target']),"flag_"+str(args.flag), "run_"+str(args.run))
            ckpt_+="/"
            if 'separate_bn' in method_args:
                if method_args["separate_bn"]:
                    ckpt_ = ckpt_ + "_"+ "separate_bn"
            if "warmup_gate" in method_args:
                if method_args["warmup_gate"]: 
                    ckpt_ = ckpt_+ "_" + "warmup_gate"
            if "RotNet" in method_args:
                if method_args["RotNet"]:
                    ckpt_ = ckpt_ + "_" + "RotNet"
        
        
        #args here is argument for linear evaluation
        ckpt = Checkpointer(
            args,
            logdir= ckpt_,
            frequency=args.checkpoint_frequency,
            keep_previous_checkpoints=False 

        )
        callbacks.append(ckpt)

    wandb_path = args.dataset + "_" + args.name + "_run_"+str(args.run)
    
    if "den_target" in method_args :
        
        wandb_path +=  "_"+str(method_args["den_target"])+"_"+"flag_"+str(args.flag)
        if 'separate_bn' in method_args:
            if method_args["separate_bn"]:
                wandb_path += "_"+ "separate_bn"
        if "warmup_gate" in method_args:
            if method_args["warmup_gate"]:   
                wandb_path += "_" + "warmup_gate"
        if "RotNet" in method_args:
            if method_args["RotNet"]:
                wandb_path += "_" + "RotNet"
            
    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=wandb_path,
            project=args.project,
            entity=args.entity,
            offline=args.offline
        )
        wandb_logger.watch(model, log="gradients", log_freq=5)
        wandb_logger.log_hyperparams(args)
    #==========================================================

    # # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    if args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        del args.resume_from_checkpoint
    else:
        ckpt_path = None

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        enable_checkpointing=False,
    )
    
    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
