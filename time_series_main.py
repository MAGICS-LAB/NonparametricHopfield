from trainers.time_series_trainer import *
import argparse
import math
import wandb
import os
import pandas as pd

def get_args():

    parser = argparse.ArgumentParser(description='Time Series Prediction with Non-parametric Hopfield Models')

    parser.add_argument('--data', type=str, choices=["ETTh1", "ETTm1", "Traffic", "WTH", "ECL", "ILI"], default='ETTh1')
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--update_steps', type=int, default=1)
    parser.add_argument('--mode', type=str, default='softmax')
    parser.add_argument('--favor_mode', type=str, default="gaussian")
    parser.add_argument('--kernel_fn', type=str, default="relu")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--in_len', type=int, default=96)
    parser.add_argument('--prob', type=float, default=0.2)
    parser.add_argument('--win_size', default=2, type=int)

    args = parser.parse_args()
    return vars(args)

if __name__ == "__main__":

    torch.set_num_threads(3)

    args = get_args()

    logs = {
        'duration':[],
        'Model':[args["mode"]],
        'Horizon Length':[args["in_len"]],
        'Prob':[args["prob"]]
    }

    in_len = args["in_len"]
    args["out_len"] = in_len
    model = args["mode"]
    data = args["data"]

    duration = 0

    df = pd.read_csv('./time_series_duration_etth1.csv')

    for _ in range(3):

        trainer = Trainer(args, logs)
        duration += trainer.time()

    duration = duration / 3

    logs['duration'].append(duration)
    df = pd.concat([pd.DataFrame(logs), df], keys=df.columns,ignore_index=True)
    df.to_csv('./time_series_duration_etth1.csv')