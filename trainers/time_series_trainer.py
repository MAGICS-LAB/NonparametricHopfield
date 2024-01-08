from datasets.time_series_data import Dataset_MTS
from torch.utils.data import DataLoader
from layers import *
from models import *
import wandb
from sklearn.metrics import r2_score
from scipy import stats

from tqdm.auto import tqdm
from fvcore.nn import FlopCountAnalysis

from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
from deepspeed.profiling.flops_profiler.profiler import get_model_profile

import time
import numpy as np


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.03):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_auc = 0

    def early_stop(self, validation_auc):
        if validation_auc < self.max_validation_auc:
            self.max_validation_loss = validation_auc
            self.counter = 0
        elif validation_auc > (self.max_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Flops:

    def __init__(self, config) -> None:
        self.config = config
    
    def _get_model(self):

        if self.config["mode"] != "favor":
            model = TSModel(out_len=self.config["out_len"],
                               d_model=self.config["d_model"],
                                n_heads=self.config["n_heads"], 
                                update_steps=self.config["update_steps"], 
                                dropout=self.config["prob"],
                                mode=self.config["mode"],
                                scale=self.config["scale"],
                                num_pattern=self.config['num_pattern'])

        elif self.config["mode"] == "favor":
            model = TSModel(out_len=self.config["out_len"],
                               d_model=self.config["d_model"],
                                n_heads=self.config["n_heads"], 
                                update_steps=self.config["update_steps"], 
                                dropout=self.config["prob"],
                                mode=self.config["mode"],
                                scale=self.config["scale"],
                                num_pattern=self.config['num_pattern'])
        return model.cuda()

    def flops_exp(self):

        self.model = self._get_model()
        prof = FlopsProfiler(self.model)
        batch_size = 32
        all_flops = all_macs = all_dur = []

        for in_len in self.config["test_in_len"]:

            dur = 0.0

            flops, macs, _ = get_model_profile(model=self.model, input_shape=(batch_size, in_len, 1), print_profile=False, as_string=False)
            all_flops.append(flops)
            all_macs.append(macs)
            for _ in range(10):
                
                x = torch.rand(batch_size, in_len, in_len).cuda()
                prof.start_profile()
                _ = self.model(x)
                dur += prof.get_total_duration()
                prof.end_profile()
            
            dur = dur / 10
            all_dur.append(dur)

        return all_flops, all_macs, all_dur

class Trainer:

    def __init__(self, config, logs) -> None:
        self.config = config
        self.logs = logs
    
    def _get_data(self):

        trainset = Dataset_MTS(
            root_path="./datasets/csv/",
            data_path=self.config["data"]+".csv",
            flag="train",
            size=[self.config["in_len"], self.config["out_len"]]
            )  

        valset = Dataset_MTS(
            root_path="./datasets/csv/",
            data_path=self.config["data"]+".csv",
            flag="val",
            size=[self.config["in_len"], self.config["out_len"]]
            )  

        testset = Dataset_MTS(
            root_path="./datasets/csv/",
            data_path=self.config["data"]+".csv",
            flag="test",
            size=[self.config["in_len"], self.config["out_len"]]
            )  

        train_loader = DataLoader(trainset, batch_size=self.config["batch_size"], shuffle=True)
        val_loader = DataLoader(valset, batch_size=self.config["batch_size"], shuffle=False)
        test_loader = DataLoader(testset, batch_size=self.config["batch_size"], shuffle=False)

        return train_loader, val_loader, test_loader

    def _get_model(self):

        if self.config["mode"] not in ["favor", "linear"]:
            print("window size", self.config["win_size"])

            model = TSModel(out_len=self.config["out_len"],
                               d_model=self.config["d_model"],
                                n_heads=self.config["n_heads"], 
                                update_steps=self.config["update_steps"], 
                                dropout=self.config["prob"],
                                mode=self.config["mode"],
                                scale=self.config["scale"],
                                win_size=self.config["win_size"]
            )

        else:

            model = TSModel(out_len=self.config["out_len"],
                               d_model=self.config["d_model"],
                                n_heads=self.config["n_heads"], 
                                update_steps=self.config["update_steps"], 
                                dropout=self.config["prob"],
                                mode=self.config["mode"],
                                scale=self.config["scale"],
                                kernel_fn=self.config["kernel_fn"]
            )

        return model.cuda()

    def _get_opt(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.config["lr"])

    def _get_cri(self):
        return torch.nn.MSELoss()

    def get_flops_model(self):

        if self.config["mode"] in ["favor", "linear"]:
            model = Favor(
                               d_model=self.config["d_model"],
                                n_heads=self.config["n_heads"], 
                                update_steps=self.config["update_steps"],
                                mode=self.config["mode"],
                                kernel_fn=self.config["kernel_fn"]
            )

        elif self.config["mode"] in ["softmax", "sparsemax"]:
            model = Hopfield(
                               d_model=self.config["d_model"],
                                n_heads=self.config["n_heads"], 
                                update_steps=self.config["update_steps"],
                                mode=self.config["mode"],
                                scale=self.config["scale"]
            )

        else:
            model = NPH(d_model=self.config["d_model"],
                        n_heads=self.config["n_heads"], 
                        update_steps=self.config["update_steps"], 
                        rand_prob=self.config["prob"],
                        mode=self.config["mode"],
                        scale=self.config["scale"],
                        win_size=self.config["win_size"])

        # self.model = model
        return model.eval().cuda()

    def flops_exp(self):

        self.model = self.get_flops_model()
        all_flops  = []
        all_dur = []
        
        with torch.no_grad():

            for in_len in [self.config["in_len"]]:

                dur = 0.0
                x = torch.rand(4, in_len, 16).cuda()
                flops = FlopCountAnalysis(self.model, (x,x))
                all_flops.append(flops.total())
                # flops, macs, _ = get_model_profile(model=self.model, input_shape=(batch_size, in_len, 1), print_profile=True, as_string=False)
                for _ in range(50):
                    
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    _ = self.model(x, x)
                    end.record()
                    torch.cuda.synchronize()
                    dur += start.elapsed_time(end)

                    torch.cuda.empty_cache()
                    
                dur = dur / 50
                all_dur.append(dur)

        return np.mean(all_flops), all_dur[0]

    def test_epoch(self, loader, prof=False):

        total_loss = 0.0
        total_cor, total_sample = 0, 0
        total_step = 0
        total_mae = 0.0
        
        total_y = []
        total_pred = []

        with torch.no_grad():
            for x, y in loader:

                total_sample += x.size(0)
                total_step += 1
                x, y = x.float().cuda(), y.float().cuda()
                pred = self.model(x)
                loss = self.cri(pred, y)
                total_loss += loss.item()
                
                total_y = total_y + y.tolist()
                total_pred = total_pred + pred.tolist()
                total_mae += self.mae(pred, y).item()

        mae, mse = total_mae/total_step, total_loss/total_step
        return mse, mae

    def train_epoch(self, loader):

        total_loss = 0.0
        total_cor, total_sample = 0, 0
        total_step = 0
        total_y = []
        total_pred = []
        total_mae = 0.0

        for x, y in tqdm(loader, ascii=True):

            total_step += 1
            total_sample += x.size(0)
            self.opt.zero_grad()
            x, y = x.float().cuda(), y.float().cuda()
            pred = self.model(x)
            loss = self.cri(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.opt.step()

            total_loss += loss.item()
            total_y = total_y + y.view(-1).tolist()
            total_pred = total_pred + pred.view(-1).tolist()

            with torch.no_grad():
                total_mae += self.mae(pred, y).item()

        mae, mse = total_mae/total_step, total_loss/total_step

        return mse, mae # , r2, pearson

    def train(self):

        train_loader, val_loader, test_loader = self._get_data()
        self.model = self._get_model()
        self.opt = self._get_opt()
        self.cri = self._get_cri()
        self.mae = nn.L1Loss()
        best_test_mse = 999
        best_result = 0
        best_mae = 0
        es = EarlyStopper()
        self.sche = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.config["epoch"], eta_min=0, last_epoch=-1, verbose=False)

        if self.config["mode"] in ["topk", "rand"]:
            print("starting training...", self.config["mode"], self.config["prob"])
        else:
            print("starting training...", self.config["mode"])

        for epoch in range(1, self.config["epoch"]+1):

            train_mse, train_mae = self.train_epoch(train_loader)
            test_mse, test_mae = self.test_epoch(val_loader)
            self.sche.step()

            print('Epoch', epoch, "Train Loss", round(train_mse, 4), "Val Loss", round(test_mse, 4) )

            if test_mse < best_test_mse:
                best_test_mse = test_mse
                best_result, best_mae = self.test_epoch(test_loader)
                print("New chechpoint...", "Test Loss", best_result)

            self.logs['train loss'].append(train_mse)
            self.logs['test loss'].append(test_mse)
            self.logs['epoch'].append(epoch)
            self.logs['model'].append(self.config['mode'])
    
        return self.logs, best_result, best_mae

    def time(self):

        train_loader, val_loader, test_loader = self._get_data()
        self.model = self._get_model()
        self.opt = self._get_opt()
        self.cri = self._get_cri()
        self.mae = nn.L1Loss()
        self.sche = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.config["epoch"], eta_min=0, last_epoch=-1, verbose=False)

        if self.config["mode"] in ["topk", "rand"]:
            print("starting training...", self.config["mode"], self.config["prob"])
        else:
            print("starting training...", self.config["mode"])

        for epoch in range(1, self.config["epoch"]+1):

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            train_mse, train_mae = self.train_epoch(train_loader)
            # train_mse, train_mae = self.test_epoch(train_loader)
            end.record()
            torch.cuda.synchronize()
            dur = start.elapsed_time(end)
    
        return dur
