# pytorch-prnet

---

## Introduction

A pytorch implement of [PRNet](https://github.com/YadiraF/PRNet), with weight transferred

## Prerequisite

- Python3
- tensorflow
- pytorch
- skimage(if you want to run with a image)

## Usage

```shell
> git clone https://github.com/liguohao96/pytorch-prnet.git
> cd pytorch-prnet
> python tf2torch.py --prnet_dir /path/to/prnet/repository
```

## Test

this will train tf_prnet and torch_prnet simultaneously on a random dataset to see if their outputs and loss are the same. (based on my experiement, they are not)

```shell
> python test/train.py --prnet_dir /path/to/prnet/repository --step
```