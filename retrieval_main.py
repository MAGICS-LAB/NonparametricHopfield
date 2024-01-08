import argparse
import json
from trainers.memory_retrieval_trainer import *
import os
import torch

def get_args():

    parser = argparse.ArgumentParser(description='Memory Retrieval benchmarks:')
    
    # model params
    parser.add_argument('--mode', default="softmax", choices=["softmax", "topk", "sparsemax", "rand", "favor", "linear"])
    parser.add_argument('--prob', default=0.5, type=float)
    parser.add_argument('--d_model', default=784, type=int)
    parser.add_argument('--favor_mode', default="gaussian", type=str)
    parser.add_argument('--kernel_fn', default='relu', type=str)
    parser.add_argument('--update_steps', default=10, type=int)
    parser.add_argument('--n_heads', default=1, type=int)
    parser.add_argument('--scale', default=0.1, type=float)

    # exp params
    parser.add_argument('--thr', default=0.1, type=float)
    parser.add_argument('--data', default="mnist", type=str)
    parser.add_argument('--method', default='half', type=str)
    parser.add_argument('--noise', default=0.05, type=float)
    parser.add_argument('--mem_size', default=10, type=int)

    args = parser.parse_args()

    return vars(args)

if __name__ == "__main__":

    torch.set_num_threads(3)
    config = get_args()

    trials = 20

    for i in range(trials):

        exp = Exp(config, i)
        exp.retrieve()
        wandb.finish()