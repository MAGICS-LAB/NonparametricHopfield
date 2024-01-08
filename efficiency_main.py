from trainers.time_series_trainer import *
import argparse
import math
import wandb
import os
import pandas as pd
import torch

def get_args():

    parser = argparse.ArgumentParser(description='Time Series Prediction with Non-parametric Hopfield Models')

    parser.add_argument('--data', type=str, choices=["ETTh1", "ETTm1", "Traffic", "WTH", "ECL", "ILI"], default='ETTh1')
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--update_steps', type=int, default=1)
    parser.add_argument('--mode', type=str, default='softmax')
    parser.add_argument('--favor_mode', type=str, default="gaussian")
    parser.add_argument('--kernel_fn', type=str, default="relu")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--in_len', type=int, default=96)
    parser.add_argument('--prob', type=float, default=0.2)

    args = parser.parse_args()
    return vars(args)

if __name__ == "__main__":


    torch.set_num_threads(3)
    logs = {
        'model':[],
        'duration':[],
        'Flops':[],
        'Sequence Length':[],
        "Prob":[]
    }
    args = get_args()
    
    in_lens = [24, 48, 96, 192, 336, 720, 1440, 2880]
    win_size = [4, 6, 8, 12, 14, 18, 30, 48]
    # for model in ["softmax", "sparsemax", "rand", "topk", "window", "favor", "linear"]:

    for model in ["softmax", "sparsemax", "rand_fast", "window", "favor", "linear"]:

    # for model in ["softmax", "rand_fast", "window"]:

        torch.cuda.empty_cache()
        args["mode"] = model
        args["win_size"] = 2

        if model in ["rand_fast"]:

            for prob in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                args["prob"] = prob

                for in_len in in_lens:

                    args["in_len"] = in_len
                    args["out_len"] = in_len

                    trainer = Trainer(args, logs)
                    flops, dur = trainer.flops_exp()

                    logs["Flops"].append(flops)
                    logs["duration"].append(dur)
                    logs["model"].append(model)
                    logs["Sequence Length"].append(in_len)
                    logs["Prob"].append(prob)
                    del trainer
        
        elif model != "window":
            for in_len in in_lens:

                prob = 0.0
                args["in_len"] = in_len
                args["out_len"] = in_len

                trainer = Trainer(args, logs)
                flops, dur = trainer.flops_exp()

                logs["Flops"].append(flops)
                logs["duration"].append(dur)
                logs["model"].append(model)
                logs["Sequence Length"].append(in_len)
                logs["Prob"].append(prob)
                del trainer

        elif model == "window":
            for i, in_len in enumerate(in_lens):
                
                w = win_size[i]
                args["win_size"] = w
                print(in_len, w)
                prob = 0.0
                args["in_len"] = in_len
                args["out_len"] = in_len

                trainer = Trainer(args, logs)
                flops, dur = trainer.flops_exp()

                logs["Flops"].append(flops)
                logs["duration"].append(dur)
                logs["model"].append(model)
                logs["Sequence Length"].append(in_len)
                logs["Prob"].append(prob)

                del trainer

    df = pd.DataFrame(logs)
    df.to_csv('./mts_cost.csv', index=False)