# Unifying Synergies between Self-supervised Learning and Dynamic Computation

This is an official Pytorch based implementation of [Unifying Synergies between Self-supervised Learning and Dynamic Computation](https://arxiv.org/pdf/2301.09164.pdf) accepted in [BMVC 2023]([https://imvipconference.github.io](https://bmvc2023.org)).

## Unification
In this work we present a novel perspective on the interplay between the SSL and DC paradigms. In particular, we show that it is feasible to simultaneously learn a dense and gated sub-network from scratch in an SSL setting without any additional fine-tuning or pruning steps. The co-evolution during pre-training of both dense and gated encoder offers a good accuracy-efficiency trade-off and therefore yields a generic and multi-purpose architecture for application-specific industrial settings. 

![](main_figure.png)

## Getting Started 

### Requirements

The main requirements of this work are:

- Python 3.8  
- PyTorch == 1.10.0  
- Torchvision == 	0.11.1
- CUDA 10.2

We recommand using conda env to setup the experimental environments.

# Install other requirements
```shell script
pip install -r requirements.txt

# Clone repo
git clone https://github.com/KrishnaTarun/SSL_DGC.git
cd ./SSL_DGC
```

### Pre-Training

```shell script

# ImageNet-100
bash bash_files/pretrain/imagent100/simsiam.sh

# CIFAR-100
bash bash_files/pretrain/cifar100/simsiam.sh

```

### KNN - Evaluation 
```shell script
# ImageNet-100
bash bash_files/knn/imagent100/knn.sh
```
###TODO citation bibtex


