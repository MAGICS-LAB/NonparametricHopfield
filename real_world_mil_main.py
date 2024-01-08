import argparse
from trainers.real_world_mil_trainer import Trainer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from datasets.loader import load_data, load_ucsb, DummyDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb


def get_args():

    parser = argparse.ArgumentParser(description='Examples of MIL benchmarks:')
    parser.add_argument('--dataset', default='fox', type=str, choices=['fox', 'tiger', 'elephant','ucsb',"musk1","musk2"])
    parser.add_argument('--mode', default='softmax', type=str, choices=['softmax', 'sparsemax', 'rand', 'topk', 'favor', 'linear'])
    parser.add_argument('--prob', default=0.2, choices=[0.2, 0.5, 0.8], type=float)
    parser.add_argument('--rs', help='random state', default=1111, type=int)
    parser.add_argument('--multiply', help='multiply features to get more columns', default=False, type=bool)
    parser.add_argument('--epoch', default=50, type=int)

    parser.add_argument('--update_steps', default=1)
    parser.add_argument('--favor_mode', default="gaussian", type=str)
    parser.add_argument('--num_pattern', default=1, type=int)
    parser.add_argument('--kernel_fn', default="relu", type=str)
    args = parser.parse_args()
    return vars(args)

class Exp:
    def __init__(self,  train_features, train_labels, testset, config):
        self.config = config
        self.train_feat, self.train_label, self.testset = train_features, train_labels, testset
    
    def run(self):
        

        wandb.init(project="MIL Real World", config=self.config)
        config = wandb.config
        # config = self.config

        trainer = Trainer(config)
        trainer.train(self.train_feat, self.train_label, self.testset)
        wandb.finish()

def main(config):

    if config["dataset"] != "ucsb":
        features, labels = load_data(config)
    else:
        features, labels = load_ucsb()
    config["feat_dim"] = features[0].shape[-1]
    config["max_len"] = max([features[id].shape[0] for id in range(len(features))])
    skf_outer = StratifiedKFold(n_splits=5, random_state=config["rs"], shuffle=True)

    for outer_iter, (train_ids, test_ids) in enumerate(skf_outer.split(features, labels)):

        train_features, train_labels = [features[id] for id in train_ids], [labels[id] for id in train_ids]
        test_features, test_labels = [features[id] for id in test_ids], [labels[id] for id in test_ids]
        testset = DummyDataset(test_features, test_labels)

        exp = Exp(train_features, train_labels, testset, config)

        sweep_config = {
            'method': 'random'
        }
        metric = {
            'name': 'val auc',
            'goal': 'maximize'   
        }

        sweep_config['metric'] = metric

        parameters_dict = {
            'lr': {
                'values': [1e-3, 1e-4, 1e-5]
                },
            'batch_size': {
                'values': [4, 8, 16]
                },
            'scale': {
                'values': [0.1, 1]
                },
            'n_heads':{
                'values':[4, 8]
                },
            'd_model':{
                'values':[128, 256, 512]
                },
            'dropout':{
                'values':[0.2, 0.5, 0.8]
                },
            'outer_iter':{
                'value':outer_iter
            },
            'weight_decay':{
                'values':[0.0, 1e-3, 1e-4]
            }
        }

        sweep_config['parameters'] = parameters_dict
        sweep_config['run_cap'] = 50
        sweep_id = wandb.sweep(sweep_config, project="Sweeps Real World MIL")
        wandb.agent(sweep_id, exp.run)
    

if __name__ == "__main__":

    config = get_args()
    main(config)