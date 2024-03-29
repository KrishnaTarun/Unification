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

import json
import os
from pathlib import Path
import time
from typing import Tuple
from xmlrpc.client import Boolean

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from solo.args.setup import parse_args_knn
from solo.methods import METHODS
from solo.utils.metrics import AverageMeter, analyse_flops

from solo.utils.classification_dataloader import (
    prepare_dataloaders,
    prepare_datasets,
    prepare_transforms,
)
from solo.utils.knn import WeightedKNNClassifier


@torch.no_grad()
def extract_features(loader: DataLoader, model: nn.Module, token=None, flop_comp=None) -> Tuple[torch.Tensor]:
    """Extract features from a data loader using a model.

    Args:
        loader (DataLoader): dataloader for a dataset.
        model (nn.Module): torch module used to extract features.

    Returns:
        Tuple(torch.Tensor): tuple containing the backbone features, projector features and labels.
    """

    model.eval()
    block_flops = AverageMeter()
    
    backbone_features, proj_features, labels = [], [], []
    for im, lab in tqdm(loader):
        im = im.cuda(non_blocking=True)
        lab = lab.cuda(non_blocking=True)
        
        if token is None:
            outs = model(im)
        else:
            flag={}
            if flop_comp: #if true set set.flag=0 i.e computation using gating mechanism
                flag["flag"]=0
                outs = model(im, flag)
            else:
                flag["flag"]=1 #skip gating
                outs = model(im, flag)        


        if flop_comp :
            batch_size = im.size()[0]
            
            flops_real = outs["flops_real"]
            flops_mask = outs["flops_mask"]
            flops_ori  = outs["flops_ori"]
            flops_conv, flops_mask, flops_ori, flops_conv1, flops_fc = analyse_flops(
                                              flops_real, flops_mask, flops_ori, batch_size)
            block_flops.update(flops_conv, batch_size)
            # pass
        backbone_features.append(outs["feats"].detach())
        # proj_features.append(outs["z"])
        labels.append(lab)
    
    backbone_features = torch.cat(backbone_features)
    # proj_features = torch.cat(proj_features)
    labels = torch.cat(labels)
    if flop_comp :
        
        model.backbone.record_flops(block_flops.avg, flops_mask, flops_ori, flops_conv1, flops_fc)
        flops = (block_flops.avg[-1]+flops_mask[-1]+flops_conv1.mean()+flops_fc.mean())/1024
        flops_per = (block_flops.avg[-1]+flops_mask[-1]+flops_conv1.mean()+flops_fc.mean())/(
                         flops_ori[-1]+flops_conv1.mean()+flops_fc.mean())*100
        flops_real = block_flops.avg[-1]+flops_mask[-1]+flops_conv1.mean()+flops_fc.mean()
        original  = flops_ori[-1]+flops_conv1.mean()+flops_fc.mean()
        reduction = 1 - (flops_per)/100 
        
        print("flops_real: {}, flops_per: {}, flops_reduction: {}, flops_orig: {}".format(flops_real.item(), flops_per.item(), reduction.item(), original.item()))
        
        log_flops = {
            "flops": flops_real.item(),
            "flops_per": flops_per.item(),
            "flops_reduction": reduction.item(),
            "flops_orig": original.item()
        }
        # pass
        
        return backbone_features, proj_features, labels, log_flops
    return backbone_features, proj_features, labels


@torch.no_grad()
def run_knn(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    k: int,
    T: float,
    distance_fx: str,
) -> Tuple[float]:
    """Runs offline knn on a train and a test dataset.

    Args:
        train_features (torch.Tensor, optional): train features.
        train_targets (torch.Tensor, optional): train targets.
        test_features (torch.Tensor, optional): test features.
        test_targets (torch.Tensor, optional): test targets.
        k (int): number of neighbors.
        T (float): temperature for the exponential. Only used with cosine
            distance.
        distance_fx (str): distance function.

    Returns:
        Tuple[float]: tuple containing the the knn acc@1 and acc@5 for the model.
    """

    # build knn
    knn = WeightedKNNClassifier(
        k=k,
        T=T,
        distance_fx=distance_fx,
    )

    # add features
    knn(
        train_features=train_features,
        train_targets=train_targets,
        test_features=test_features,
        test_targets=test_targets,
    )

    # compute
    acc1, acc5 = knn.compute()

    # free up memory
    del knn

    return acc1, acc5


def main():
    # torch.set_deterministic(True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    args = parse_args_knn()
    t1 = time.time()
    
    print(args.pretrained_checkpoint_dir)

    #version type
    ver_type = args.pretrained_checkpoint_dir.split("/")[-1]
    ver_arr  = args.pretrained_checkpoint_dir.split("/")
    print(ver_type)

    if not os.path.isdir(args.pretrained_checkpoint_dir):
        print("No such checkpoint folder")
        exit()

    # build paths
    ckpt_dir = Path(args.pretrained_checkpoint_dir)
    # print(ckpt_dir, len(os.listdir(ckpt_dir)))
    args_path = ckpt_dir / "args.json"

    N = len(os.listdir(ckpt_dir)) - 1 #remove json
    
    
    # # NOTE neglect these for moments
    ckpt_path = [os.path.join(ckpt_dir,ckpt) for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")]
    print(ckpt_path) 
    # # load arguments
    if os.path.isfile(args_path):
        with open(args_path) as f:
            method_args = json.load(f)

    else:
        print("No json found")
        print("==================================")
        exit()
    


    #just to avoid exception error with previous scripts
    #this might change later
    #keep single json but add new key-value pair
    json_path = "./"+ ver_type +"_knn.json"
    key_name = str(ver_type)
    if 'separate_bn' in method_args:
        if method_args["separate_bn"]:
            key_name += "_"+ "separate_bn"
    if "warmup_gate" in method_args:
        if method_args["warmup_gate"]:   
            key_name += "_" + "warmup_gate"
    if "RotNet" in method_args:
        if method_args["RotNet"]:
            key_name += "_" + "RotNet"
    print("==============================")    
    print(key_name)
    print("==============================")
    # exit()
   
    # create a json to store results as knn.json
    if os.path.isfile(json_path):
        with open(json_path) as f:
            print("Loading json")
            knn = json.load(f)
    else:
        knn = {}
    
    
    if "den_target" in method_args:
        assert (str(method_args["den_target"])==ver_type)
        
        print("exist")
        # flop_comp = True
        try:
            # print("hello")
            _ = knn[key_name]
        except :
        
            knn[key_name] = {}
    else:
        # flop_comp = False
        try:
            _ = knn[str(method_args["name"])]
        except:
            knn[str(method_args["name"])] = {}

        pass

    
    # prepare data
    # #TODO may take this out of the loop
    _, T = prepare_transforms(args.dataset)
    train_dataset, val_dataset = prepare_datasets(
        args.dataset,
        T_train=T,
        T_val=T,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        download=True,
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    print(method_args["method"])
    # build the model
    for md_pth in ckpt_path: #iter over the list

        print("Loading {}".format(md_pth))
        eph = md_pth.split("/")[-1]
        # print(eph)
        model = METHODS[method_args["method"]].load_from_checkpoint(
            md_pth, strict=True, **method_args
        )
        # print(model)
        model.cuda()
        print(args.data_dir, args.train_dir, args.val_dir )

        if args.token==None:

            train_features_bb, _, train_targets = extract_features(train_loader, model, args.token, flop_comp=None)
            # train_features = {"backbone": train_features_bb, "projector": _}
            test_features_bb, _, test_targets   = extract_features(val_loader, model,  args.token, flop_comp=None)   
            
        else: #two cases use gating or skip gating:
            #use gating set flop_comp to true
            if not args.flag: #use gating
                flop_comp = True
                train_features_bb, _, train_targets, _   = extract_features(train_loader, model, args.token, flop_comp)
                test_features_bb, _, test_targets, flops = extract_features(val_loader, model,args.token, flop_comp)
            else:
                flop_comp =False
                train_features_bb, _, train_targets = extract_features(train_loader, model, args.token, flop_comp)
                test_features_bb,  _, test_targets  = extract_features(val_loader, model,args.token, flop_comp)


        train_features = {"backbone": train_features_bb}
        test_features =  {"backbone": test_features_bb}
        
        del _
        
        #NOTE running single iteration saving result in json might change for all combinations
        # run k-nn for all possible combinations of parameters
        for feat_type in args.feature_type:
            print(f"\n### {feat_type.upper()} ###")
            print(args.k)
            for k in args.k:
                for distance_fx in args.distance_function:
                    temperatures = args.temperature if distance_fx == "cosine" else [None]
                    for T in temperatures:
                        print("---")
                        print(f"Running k-NN with params: distance_fx={distance_fx}, k={k}, T={T}...")
                        acc1, acc5 = run_knn(
                            train_features=train_features[feat_type],
                            train_targets=train_targets,
                            test_features=test_features[feat_type],
                            test_targets=test_targets,
                            k=k,
                            T=T,
                            distance_fx=distance_fx,
                        )
                        print(f"Result: acc@1={acc1}, acc@5={acc5}")
                    if "den_target" in method_args:
                        try:
                            _ = knn[key_name]["token_with_flag_"+str(args.flag)]
                        except:
                            knn[key_name]["token_with_flag_"+str(args.flag)]={}

                        knn[key_name]["token_with_flag_"+str(args.flag)][eph]=acc1
                        if not args.flag:
                            knn[key_name]["token_with_flag_"+str(args.flag)]["flops"] = flops
                    else:
                        knn[str(method_args["name"])][eph] = acc1

    #                 # print(knn)

        del model
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(knn, f, ensure_ascii=False, indent=4)
        torch.cuda.empty_cache()

        # break
    print("time:{}".format(time.time()-t1))


if __name__ == "__main__":
    main()
