# Energy-Efficient Federated Learning via Dynamic Computation Offloading

> Pytorch implementation for energy-efficient federated learning via dynamic computation offloading.

## Installation

First check that the requirements are satisfied:</br>
Python 3.6</br>
torch 1.2.0</br>
torchvision 0.4.0</br>
numpy 1.18.1</br>
sklearn 0.20.0</br>
matplotlib 3.1.2</br>
Pillow 4.1.1</br>

The next step is to clone the repository:
```bash
git clone https://github.com/pliang279/LG-FedAvg.git
```

## Data

We run experiments on MNIST ([link](http://yann.lecun.com/exdb/mnist/)) and CIFAR10 ([link](https://www.cs.toronto.edu/~kriz/cifar.html)).

## FedAvg

Results can be reproduced running the following:

#### MNIST
> python main_fed_energy.py --dataset mnist --model mlp --num_classes 10 --epochs 350  --lr 0.05 --num_users 100  --shard_per_user 10  --frac 0.5 --local_ep 1 --local_bs 10  --results_save run1 --iid --server_capacity 50

#### CIFAR10 
> python main_fed_energy.py --dataset cifar10 --model cnn --num_classes 10 --epochs 2000 --lr 0.1 --num_users 100 --shard_per_user 10 --frac 0.5 --local_ep 1 --local_bs 50 --results_save run1 -- iid --server_capacity 50

# Acknowledgements

This codebase was adapted from https://github.com/pliang279/LG-FedAvg.
