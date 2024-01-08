import torch
import wandb
from datasets.loader import load_data, DummyDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from models import *


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.03):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_auc = 0

    def early_stop(self, validation_auc):
        if validation_auc > self.max_validation_auc:
            self.max_validation_loss = validation_auc
            self.counter = 0
        elif validation_auc < (self.max_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Trainer:

    def __init__(self, config) -> None:

        self.config = config
        # run = wandb.init(config=self.config)

    def _get_model(self):

        if self.config["mode"] != "favor":
            model = MILModel(input_size=self.config["feat_dim"],
                               d_model=self.config["d_model"],
                                n_heads=self.config["n_heads"], 
                                update_steps=self.config["update_steps"], 
                                prob=self.config["prob"],
                                dropout=self.config["dropout"],
                                mode=self.config["mode"],
                                scale=self.config["scale"],
                                num_pattern=self.config['num_pattern'])

        elif self.config["mode"] in ["favor", "linear"]:
            model = MILModel(input_size=self.config["feat_dim"],
                               d_model=self.config["d_model"],
                                n_heads=self.config["n_heads"], 
                                update_steps=self.config["update_steps"], 
                                dropout=self.config["dropout"],
                                mode=self.config["mode"],
                                scale=self.config["scale"],
                                num_pattern=self.config['num_pattern'],
                                favor_mode=self.config["favor_mode"],
                                kernel_fn=self.config["kernel_fn"])

        return model.cuda()

    def _get_opt(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])

    def _get_cri(self):
        return F.binary_cross_entropy_with_logits

    def test_epoch(self, loader):

        total_loss = 0.0
        total_cor, total_sample = 0, 0
        total_step = 0
        total_roc = 0.0

        with torch.no_grad():
            for x, y, mask in loader:

                total_step += 1
                total_sample += x.size(0)
                x, y, mask = x.cuda(), y.cuda(), mask.cuda()
                pred = self.model(x, mask=mask)
                loss = self.cri(input=pred.float(), target=y.float())

                output = (torch.sigmoid(pred)>0.5).float()
                total_cor += (output == y).float().sum()
                total_loss += loss.item()
                
                roc = roc_auc_score(y.squeeze().detach().cpu(), pred.sigmoid().squeeze().detach().cpu())
                total_roc += roc
        
        return total_loss/total_step, total_cor/total_sample, total_roc/total_step

    def train_epoch(self, loader):

        '''
        need mask
        '''

        total_loss = 0.0
        total_cor, total_sample = 0, 0
        total_step = 0

        for x, y, mask in loader:

            total_step += 1
            total_sample += x.size(0)

            self.opt.zero_grad()
            x, y, mask = x.cuda(), y.cuda(), mask.cuda()
            pred = self.model(x, mask=mask)
            loss = self.cri(input=pred.float(), target=y.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0, norm_type=2)
            self.opt.step()

            output = (torch.sigmoid(pred)>0.5).float()
            total_cor += (output == y).float().sum()
            total_loss += loss.item()

        return total_loss/total_step, total_cor/total_sample

    def _get_data(self, train_features, train_labels, testset):

        skf_inner = StratifiedKFold(n_splits=5, random_state=self.config["rs"], shuffle=True)
        train_subset_ids, val_subset_ids = next(skf_inner.split(train_features, train_labels))
        train_subset_features, train_subset_labels = [train_features[id] for id in train_subset_ids] \
            , [train_labels[id] for id in train_subset_ids]
        val_subset_features, val_subset_labels = [train_features[id] for id in val_subset_ids] \
            , [train_labels[id] for id in val_subset_ids]
        train_subset, val_subset = DummyDataset(train_subset_features, train_subset_labels) \
            , DummyDataset(val_subset_features, val_subset_labels)

        trainloader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=int(self.config["batch_size"]),
            shuffle=True,
            num_workers=8,
            collate_fn=testset.collate
        )
        valloader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=len(val_subset),
            shuffle=True,
            num_workers=8,
            collate_fn=testset.collate
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=len(testset),
            shuffle=False,
            num_workers=8,
            collate_fn=testset.collate
        )
        
        return trainloader, valloader, testloader

    def train(self, train_features, train_labels, testset):

        train_loader, val_loader, test_loader = self._get_data(train_features, train_labels, testset)
        self.model = self._get_model()
        self.opt = self._get_opt()
        self.cri = self._get_cri()
        early_stopper = EarlyStopper()

        self.sche = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.config["epoch"], eta_min=0, last_epoch=-1, verbose=False)

        best_val_auc = -1

        for epoch in range(1, self.config["epoch"]+1):

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_auc = self.test_epoch(val_loader)

            # print("train loss", train_loss)
            # print("val loss", val_loss)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                test_loss, test_acc, test_auc = self.test_epoch(test_loader)
        
            # if early_stopper.early_stop(val_auc):
            #     break

            self.sche.step()

            wandb.log({
                "epoch": epoch,
                "train loss": train_loss,
                "train acc": train_acc*100,
                "val loss": val_loss,
                "val acc": val_acc*100,
                "val auc": val_auc
                })

        wandb.log({
            "test auc": test_auc
        })