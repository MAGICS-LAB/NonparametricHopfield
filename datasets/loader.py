import numpy as np
import scipy.io
import os
import pickle
import pandas as pd
# import pandas as pd
import sklearn.model_selection
from sklearn.model_selection import StratifiedKFold

import torch
import torch.utils.data
from torchvision import datasets, transforms

from .loader_utils import *
import random

def load_mnist(batch_size,norm_factor=1):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./mnist_data', train=True,
                                            download=True, transform=transform)
    print("trainset: ", trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True)
    print("trainloader: ", trainloader)
    trainset = list(iter(trainloader))

    testset = datasets.MNIST(root='./mnist_data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True)
    testset = list(iter(testloader))
    return trainset, testset



class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()

        self.x = x
        self.y = y
        for i in range(len(self.y)):
            if self.y[i] == -1:
                self.y[i] = 0.0

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        batch_x = self.x[idx] # (bag_size, feat_dim)
        batch_y = self.y[idx]

        batch_x = torch.tensor(batch_x)
        batch_y = torch.tensor(batch_y)

        return batch_x, batch_y

    def collate(self, batch):

        x = [x for x,y in batch]
        y = [y for x,y in batch]

        pad_batch_x, mask_x = self.padding(x)

        return pad_batch_x, torch.stack(y, dim=0), mask_x

    def padding(self, batch):

        max_bag_len = max([len(xi) for xi in batch]) # (batch_size, bag_size, feat_dim)
        feat_dim = batch[0].size(-1)
        # print(feat_dim, max_bag_len)

        batch_x_tensor = torch.zeros((len(batch), max_bag_len, feat_dim))
        # mask_x = torch.zeros((len(batch), max_bag_len), dtype=torch.uint8)
        mask_x = torch.ones((len(batch), max_bag_len), dtype=torch.uint8)

        for i in range(len(batch)):
            bag_size = batch[i].size(0)
            batch_x_tensor[i, :bag_size] = batch[i]
            mask_x[i][:bag_size] = 0.0
            # mask_x[i][:bag_size] = 1.0
        mask_x = mask_x.to(torch.bool)
        return batch_x_tensor, mask_x

def load_data(args):
    features = []
    labels = []
    dataset = scipy.io.loadmat(f'./datasets/mil_datasets/{args["dataset"]}_100x100_matlab.mat')  # loads fox dataset
    instance_bag_ids = np.array(dataset['bag_ids'])[0]
    instance_features = np.array(dataset['features'].todense())
    # print(instance_features[0].shape)
    if args['multiply']:
        instance_features = multiply_features(instance_features)

    instance_labels = np.array(dataset['labels'].todense())[0]
    bag_features = into_dictionary(instance_bag_ids,
                                   instance_features)  # creates dictionary whereas key is bag and values are
    bag_labels = into_dictionary(instance_bag_ids,
                                 instance_labels)  # creates dictionary whereas key is bag and values are instance
    for i in range(1, len(bag_features) + 1):  # goes through whole dataset
        features.append(np.array(bag_features.pop(i)))
        labels.append(max(bag_labels[i]))
    return features, labels

def load_ucsb():
    
    '''
    This function Returns trainset and testset
    '''

    def load_data(filepath):
        df = pd.read_csv(filepath, header=None)
        
        bags_id = df[1].unique()
        bags = [df[df[1]==bag_id][df.columns.values[2:]].values.tolist() for bag_id in bags_id]
        y = df.groupby([1])[0].first().values
        bags = [np.array(b) for b in bags]
        return bags, np.array(y)

        # split train and test data
        # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(bags, y, test_size=0.2)
        
        # trainset, testset = DummyDataset(X_train, y_train), DummyDataset(X_test, y_test)

        # return trainset, testset

    current_file = os.path.abspath(os.path.dirname(__file__))
    return load_data(current_file + '/csv/ucsb_breast_cancer.csv')

def get_dataset(args, dataset='fox'):
    """
    Loads and batches fox dataset into feature and bag label lists
    :return: list(features), list(bag_labels)
    """
    if args["multiply"]:
        filepath = './datasets/mil_datasets/{}_dataset.pkl'.format(args["dataset"])
    else:
        filepath = './datasets/mil_datasets/{}_original_dataset.pkl'.format(args["dataset"])
    # if (os.path.exists(filepath)):
    #     print('Dataset loaded')
    #     with open(filepath, 'rb') as dataset_file:
    #         dataset =  pickle.load(dataset_file)
    #         return dataset
    # else:
    dataset = Dataset(args, dataset)
    # print('Dataset loaded')
    file = open(filepath, 'wb')
    pickle.dump(dataset, file)
    return dataset

class Dataset():
    def __init__(self, args, dataset='fox'):
        """
        Loads and batches elephant dataset into feature and bag label lists
        :return: list(features), list(bag_labels)
        """
        self.rs = args["rs"] # random state
        self.features = []
        self.bag_labels = []
        dataset = scipy.io.loadmat(f'./datasets/mil_datasets/{dataset}_100x100_matlab.mat')  # loads fox dataset
        instance_bag_ids = np.array(dataset['bag_ids'])[0]
        instance_features = np.array(dataset['features'].todense())
        # print(instance_features[0].shape)
        if args["multiply"]:
            instance_features = multiply_features(instance_features)

        instance_labels = np.array(dataset['labels'].todense())[0]
        bag_features = into_dictionary(instance_bag_ids,
                                       instance_features)  # creates dictionary whereas key is bag and values are
        bag_labels = into_dictionary(instance_bag_ids,
                                     instance_labels)  # creates dictionary whereas key is bag and values are instance
        for i in range(1, 201):  # goes through whole dataset
            self.features.append(np.array(bag_features.pop(i)))
            self.bag_labels.append(max(bag_labels[i]))
        self.random_shuffle()

    def random_shuffle(self):
        self.features, self.bag_labels = shuffle_dataset(self.features, self.bag_labels, self.rs)
        x_train, x_test, y_train, y_test = train_test_split(self.features, self.bag_labels)
        self.training_data = x_train
        self.testing_data = x_test
        self.training_labels = y_train
        self.testing_labels = y_test

    def return_training_set(self):

        trainset = DummyDataset(self.training_data, self.training_labels)
        return trainset

    def return_testing_set(self):
        testset = DummyDataset(self.testing_data, self.testing_labels)
        return testset

    def return_dataset(self):
        fullset = DummyDataset(self.features, self.bag_labels)
        return fullset

class MnistBags(torch.utils.data.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=2, num_bag=250, seed=1, train=True):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState(seed)

        self.num_in_train = 60000
        self.num_in_test = 10000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        if self.train:
            loader = torch.utils.data.DataLoader(datasets.MNIST('../datasets',
                                                          train=True,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_train,
                                           shuffle=False)
        else:
            loader = torch.utils.data.DataLoader(datasets.MNIST('../datasets',
                                                          train=False,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_test,
                                           shuffle=False)

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        for i in range(self.num_bag):
            bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            if bag_length < 1:
                bag_length = 1

            if self.train:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_train, bag_length))
            else:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_test, bag_length))

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag == self.target_number

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]

        return bag, label

def cifar_mem_retrieve(memory_size):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    
    images = []
    for i, (x, y) in enumerate(trainset):
        images.append(x)

    images = random.sample(images, memory_size)

    return images

def mnist_mem_retrieve(memory_size):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.1307,), std = (0.3081,))
        ]
    )
    
    trainset = datasets.MNIST(
        '../datasets',
        train=True,
        download=True,
        transform=transform
    )

    images = []
    for i, (x, y) in enumerate(trainset):
        images.append(x)

    images = random.sample(images, memory_size)

    return images

def imagenet_mem_retrieve(memory_size):
    pass
