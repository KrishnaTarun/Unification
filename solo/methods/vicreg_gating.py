"""
    implements teacher-student setting 
"""

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

import argparse
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from solo.losses.vicreg import vicreg_loss_func
from solo.methods.base_gating import BaseMethod

"""
Vanilla VicReg
"""

class VICReg(BaseMethod):
    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        sim_loss_weight: float,
        var_loss_weight: float,
        cov_loss_weight: float,
        **kwargs
    ):
        """Implements VICReg (https://arxiv.org/abs/2105.04906)

        Args:
            proj_output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            sim_loss_weight (float): weight of the invariance term.
            var_loss_weight (float): weight of the variance term.
            cov_loss_weight (float): weight of the covariance term.
        """

        super().__init__(**kwargs)
        # print(kwargs['separate_bn'], kwargs['RotNet'])
        #done for previous versions without such keys
        if "RotNet" in kwargs:
            self.rotation = kwargs['RotNet']
        else:
            self.rotation = False

        if "inv" in kwargs:
            self.inv = kwargs["inv"]
        else:
            self.inv = False

        
        
        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight
        
        self.rot_weight = 0.2
        
        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        if self.rotation:
            self.rotation_projector = nn.Sequential(nn.Linear(512, 2048),
                                            nn.LayerNorm(2048),
                                            nn.ReLU(inplace=True),  # first layer
                                            nn.Linear(2048, 2048),
                                            nn.LayerNorm(2048),
                                            nn.Linear(2048, 4))  # output layer



    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(VICReg, VICReg).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("vicreg")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=2048)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--sim_loss_weight", default=25, type=float)
        parser.add_argument("--var_loss_weight", default=25, type=float)
        parser.add_argument("--cov_loss_weight", default=1.0, type=float)
        #to choose if, want to train with invariance loss or not
        parser.add_argument("--inv", action="store_true")# if inv ignore invariance calculation


        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]

        if self.rotation:
            extra_learnable_params.append({"params":self.rotation_projector.parameters(),"lr":0.1})
        print(extra_learnable_params)
        # exit()

        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        return {**out, "z": z}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for VICReg reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of VICReg loss and classification loss.
        """
        out = super().training_step(batch, batch_idx)
        # class_loss = out["loss"]
        feats1, feats2 = out["feats"] #feat1: student, #feat2: teacher

        

        z1 = self.projector(feats1)#student
        z2 = self.projector(feats2)#teacher

        if self.inv:#if true that means ignore self.sim_loss_weight
            self.sim_loss_weight = 0.0

        # ------- vicreg loss ----------N
        vicreg_loss = vicreg_loss_func(
            z1,
            z2,
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        self.log("train_vicreg_loss", vicreg_loss, on_step=True, on_epoch=True, sync_dist=True)
        
        #--------rotation loss-----------------
        #NOTE no rotated labels yet
        if self.rotation:
           rotated_labels = out["rotnet_labels"].squeeze()
           logits_z2 = self.rotation_projector(feats2)
        #    print(rotated_labels.shape)
           
           rot_loss = torch.nn.functional.cross_entropy(logits_z2, rotated_labels.to(logits_z2.get_device()), reduction="mean") 
        #    print(rot_loss)
           self.log("rotation_loss", rot_loss, on_step=True, on_epoch=True, sync_dist=True)
           vicreg_loss += self.rot_weight*rot_loss 



        if self.warmup_gate:
            if (self.current_epoch + 1) <= self.warmup_epochs:
                """
                    taken care above
                """
                return vicreg_loss
        
        # ------- collect sparsity loss ---------
        sp1_l = out["rloss"][0].mean() + out["bloss"][0].mean()
        # sp2_l = out["rloss"][1].mean() + out["bloss"][1].mean()
        # ----------------------------------------

        metrics = {
            # "train_vicreg_loss": vicreg_loss,
            "train_spar_loss":  out["rloss"][0].mean(),
            "train_bound_loss": out["bloss"][0].mean(),
            "train_total_sparsity_loss": sp1_l,   
        }
        self.log_dict(metrics, on_step=True, on_epoch=False, sync_dist=True)

        return vicreg_loss + sp1_l
