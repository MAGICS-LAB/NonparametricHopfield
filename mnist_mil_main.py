import argparse
import json
from trainers.mnist_mil_trainer import *
import os
import torch

def get_args():

    parser = argparse.ArgumentParser(description='MNIST MIL benchmarks:')

    parser.add_argument("--project_name", default="MNIST-MIL")
    parser.add_argument('--wandb', default=False, type=bool)

    # Model params
    parser.add_argument('--mode', default="softmax", choices=["softmax", "topk", "sparsemax", "rand", "favor", "linear"])
    parser.add_argument('--prob', default=0.5, type=float)
    parser.add_argument('--d_model', default=256, type=int)
    parser.add_argument('--input_size', default=784, type=int)
    parser.add_argument('--model', default="pooling", type=str)
    parser.add_argument('--num_pattern', default=2, type=int)
    parser.add_argument('--n_heads', default=4, type=int)
    parser.add_argument('--scale', default=0.1)
    parser.add_argument('--favor_mode', default="gaussian", type=str)
    parser.add_argument('--kernel_fn', default='relu', type=str)
    parser.add_argument('--update_steps', default=1, type=int)

    # Training params
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--seed', default=1111, type=int)

    # Data params
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--train_size', default=2000, type=int)
    parser.add_argument('--test_size', default=500, type=int)
    parser.add_argument('--pos_per_bag', default=1, type=int)
    parser.add_argument('--bag_size', default=10, type=int)
    parser.add_argument('--tgt_num', default=9, type=int)


    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":

    torch.set_num_threads(3)
    config = get_args()
    trails = 1

    if config["bag_size"] == 100:
        config["num_pattern"] = 4

    bag_size = [5, 10, 20, 30, 50, 80, 100, 200, 500]
    models = ["softmax", "sparsemax"]    

    for t in range(trails):
        trainer = Trainer(config, t)
        trainer.train()
