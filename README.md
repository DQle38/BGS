<div align="center">

# Continual Learning in the Presence of Spurious Correlations: Analyses and a Simple Baseline
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.8.5-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8.2-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://numpy.org/"><img alt="Numpy" src="https://img.shields.io/badge/-Numpy 1.19.1-013243?style=for-the-badge&logo=numpy&logoColor=white"></a>


</div>

## 📌&nbsp;&nbsp;Introduction

This is the official repository for ICLR 2024 paper: [**Continual Learning in the Presence of Spurious Correlations: Analyses and a Simple Baseline**](https://openreview.net/forum?id=3Y7r6xueJJ). Our paper investigates the bias transfer in continual learning (CL) and provide a simple baseline, BGS. We hope our study can serve
as a stepping stone in developing more advanced bias-aware CL methods. Thanks for your interest!

## Installation
We highly recommend you to use our conda environment.
```bash
# clone project   
git clone https://github.com/DQle38/BGS.git

# install project   
cd BGS
conda env create -f bgs_env.yaml
conda activate bgs
```

## Dataset
Split CIFAR-100S and CelebA are constructed based on torchvision dataset
([CIFAR100](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR100.html#cifar100), 
[CelebA](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR100.html#cifar100)). 
For Split ImageNet-100C, we employ [ImageNet100](https://www.kaggle.com/datasets/ambityga/imagenet100) dataset. To construct Split ImageNet-100C, you need to download the dataset from the link, and reorganize it in the structure below.
```plaintext
├──imagenet100/
    ├── train/
    │   ├── n01440764
    │   │       :
    │   └── n02077923
    └── val/
        ├── n01440764
        │       :
        └── n02077923
```

## Training
Our codes utilize [WandB](https://wandb.ai/) for logger.
To check the results, we recommend using such tools. Please refer to [WandB Quickstart](https://docs.wandb.ai/quickstart).
```python
# please change the below lines in the "main.py" with your project and entity name.
wandb.init(
        project='none',
        entity='none',
)
```

### For the investigation of bias transfer in two-task CL / CL with a longer sequence (10 tasks) 


```bash
# sample commands for the bias level of T1: 0 on Split CIFAR-100S
$ python3 main.py --skew-ratio 0.5 0.5 --trainer vanilla --bs 256 --epochs 70 --n-tasks 2 --start-task 0 --dataset split_cifar_100s --lr 0.001 --decay 0.01 --continual task --optim adamw
$ python3 main.py --skew-ratio 0.5 0.5 --trainer freezing --bs 256 --epochs 70 --n-tasks 2 --start-task 0 --dataset split_cifar_100s --lr 0.001 --decay 0.01 --continual task --optim adamw
$ python3 main.py --skew-ratio 0.5 0.5 --trainer lwf --lamb 1 --bs 256 --epochs 70 --n-tasks 2 --start-task 0 --dataset split_cifar_100s --lr 0.001 --decay 0.01 --continual task --optim adamw

# sample commands for the bias level of T1: 6 on Split CIFAR-100S (forward transfer)
$ python3 main.py --skew-ratio 0.99 0.5 --trainer vanilla --bs 256 --epochs 70 --n-tasks 2 --start-task 0 --dataset split_cifar_100s --lr 0.001 --decay 0.01 --continual task --optim adamw
$ python3 main.py --skew-ratio 0.99 0.5 --trainer freezing --bs 256 --epochs 70 --n-tasks 2 --start-task 0 --dataset split_cifar_100s --lr 0.001 --decay 0.01 --continual task --optim adamw
$ python3 main.py --skew-ratio 0.99 0.5 --trainer lwf --lamb 1 --bs 256 --epochs 70 --n-tasks 2 --start-task 0 --dataset split_cifar_100s --lr 0.001 --decay 0.01 --continual task --optim adamw

# sample commands for the bias level of T2: 6 on Split CIFAR-100S (backward transfer)
$ python3 main.py --skew-ratio 0.5 0.99 --trainer vanilla --bs 256 --epochs 70 --n-tasks 2 --start-task 0 --dataset split_cifar_100s --lr 0.001 --decay 0.01 --continual task --optim adamw
$ python3 main.py --skew-ratio 0.5 0.99 --trainer lwf --lamb 1 --bs 256 --epochs 70 --n-tasks 2 --start-task 0 --dataset split_cifar_100s --lr 0.001 --decay 0.01 --continual task --optim adamw
```
To control the bias level of the two tasks in the above commands, please vary the argument "--skew-ratio".
```bash
--skew-ratio <skewness of the first task> <skewness of the second task>
```
To check the results, we provide an example to load and plot them. Please refer to "example/cifar_bias_transfer.ipynb".

For the longer sequence experiment, set "--n-tasks" to 10 and add more skew-ratio after "--skew-ratio" to set the skewness of each task in the sequence as below.
```bash
# arguments for the forward transfer of bias
--n-tasks 10 --skew-ratio 0.99 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 # the first task is biased
--n-tasks 10 --skew-ratio 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 # the first task is not biased
```

Hyperparmaeters for CL baselines:
* "--lamb": for regularization-based methods
* "--buffer-ratio": for rehearsal-sed methods
* "--prune-ratio": for PackNet

### For the accumulation of the bias
For the accumulation experiment, utilize the "--accumulation", "--n-biased" arguments.
```bash
--accumulation --n-biased <the number of biased tasks>
# a sample command. please train in an interactive way for each method
$ python3 main.py --trainer vanilla --n-biased 2 --accumulation --bs 256 --epochs 70 --n-tasks 5 --start-task 0 --dataset split_cifar_100s --lr 0.001 --decay 0.01 --continual task --optim adamw

```
### For the comparison of CL baselines
Please train a model in an interactive way according to each setting.
```bash
# a sample script for Split CIFAR-100S
bash scripts/cifar_performance_comparison.sh
```

## Citation   
```
@article{lee2024bgs,
  title={Continual Learning in the Presence of Spurious Correlations: Analyses and a Simple Baseline},
  author={Lee, Donggyu and Jung, Sangwon and Moon, Taesup},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
