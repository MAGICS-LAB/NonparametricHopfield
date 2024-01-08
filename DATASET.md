# Usage of MIL Datasets

## Immune Repertoire Classification

This dataset contains 4 types of variations

* standard CMV
* CMV with implant signals
* Simulated
* LSTM generated

**Please do not use the cmv_implanted_dataset yet, the filelink now is missing.** <br>
**The simulated dataset is still under testing (due to a very large file size)** <br>
**To download the lstm_generated and CMV dataset, please run `python3 download_dataset.py`**

```python
from deeprc.dataset_readers import make_dataloaders
from deeprc.predefined_datasets import *

batch_size = 4

# For the CMV dataset
task_definition, train_loader, train_loader_fixed, val_loader, test_loader = cmv_dataset(dataset_path='./datasets/cmv/', batch_size=batch_size)

# For the CMV + Implant Signal  dataset
task_definition, train_loader, train_loader_fixed, val_loader, test_loader = cmv_implanted_dataset(dataset_path='./datasets/cmv_implanted/', batch_size=batch_size)

# For the Simulated dataset

task_definition, train_loader, train_loader_fixed, val_loader, test_loader = simulated_dataset(dataset_path='./datasets/simulated/', batch_size=batch_size)

# For the lstm generated dataset

task_definition, train_loader, train_loader_fixed, val_loader, test_loader = lstm_generated_dataset(dataset_path='./datasets/lstm/', batch_size=batch_size)

for x, y in train_loader:
    pass

```

## MIL Benchmarks

This section contains 3 types of datasets
* Corel Datasets (elephant, tiger, fox)
* MNIST
* UCSB Breast Cancer

### Corel Datasets

#### Download Dataset

`cd datasets/mil_datasets/` <br>
`wget http://www.cs.columbia.edu/~andrews/mil/data/MIL-Data-2002-Musk-Corel-Trec9-MATLAB.tgz` <br>
`tar zxvf ./MIL-Data-2002-Musk-Corel-Trec9-MATLAB.tg` <br>

```python

from datasets import loader
import argparse

parser = argparse.ArgumentParser(description='Examples of MIL benchmarks:')
parser.add_argument('--dataset', default='fox', type=str, choices=['fox', 'elephant', 'tiger'])
parser.add_argument('--rs', help='random state', default=1111, type=int)
parser.add_argument('--multiply', help='multiply features to get more columns', default=False, type=bool)

args = parser.parse_args()

dataset = loader.get_dataset(args, args.dataset)
trainset = dataset.return_training_set()
trainloader = DataLoader(trainset, batch_size=4, collate_fn=trainset.collate)

for x, y, mask in trainloader:
    pass

# This is a numpy array dataset

```

### UCSB Breast Cancer
```python
from datasets import loader
import argparse

parser = argparse.ArgumentParser(description='Examples of MIL benchmarks:')
parser.add_argument('--dataset', default='ucsb', type=str, choices=['ucsb'])
args = parser.parse_args()

trainset, testset = loader.load_ucsb()

train_loader = DataLoader(trainset, batch_size=2, collate_fn=trainset.collate)

for x, y, mask in train_loader:
    pass

# x : (batch_size, max_bag_size, feature_dim)
# mask : (batch_size, max_bag_size)

# This is a numpy array dataset
```

### MNIST

This is still under development

```python

from datasets import loader
import argparse
import torch


parser = argparse.ArgumentParser(description='Examples of MIL benchmarks:')
parser.add_argument('--dataset', default='mnist', type=str, choices=['mnist'])
args = parser.parse_args()

train_loader = torch.utils.data.DataLoader(loader.MnistBags(target_number=9,
                                                mean_bag_length=10,
                                                var_bag_length=2,
                                                num_bag=100,
                                                seed=98,
                                                train=True),
                                                batch_size=batch_size,
                                                shuffle=False, **kwargs)

test_loader = torch.utils.data.DataLoader(loader.MnistBags(target_number=9,
                                                mean_bag_length=10,
                                                var_bag_length=2,
                                                num_bag=10,
                                                seed=98,
                                                train=False),
                                                batch_size=batch_size,
                                                shuffle=False, **kwargs)

# This is a torch data loader

```