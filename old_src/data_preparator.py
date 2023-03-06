import json
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader
import os
from numpy import save, load


class DataProcess(object):
    def __init__(self, dataset_path, out_dir, batch_size):
        self.dataset_path, self.out_dir = self.check_path(dataset_path, out_dir)
        self.batch_size = batch_size

    def check_path(self, dataset_path, out_dir):
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        return dataset_path, out_dir

    def load_datasets(self, dataset_name, cifar_type=None):
        normalization_map = {'MNIST': [(0.5, 0.5), (0.5, 0.5)], 'CIFAR': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
                             'SVHN': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]}
        dataset_name = dataset_name if cifar_type is None else f'{dataset_name}{cifar_type}'
        train, test, root = self.get_dataset(dataset_name, normalization_map[dataset_name])
        return train, test, root

    def get_dataset(self, dataset_name, norm_map):
        root = os.path.join(self.dataset_path, dataset_name)
        print(f"Dataset is {dataset_name}")
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(norm_map[0], norm_map[1])]
        )
        if dataset_name == 'CIFAR10':
            train, test = self.cifar10(root, transform)
        elif dataset_name == 'CIFAR100':
            train, test = self.cifar100(root, transform)
        elif dataset_name == 'MNIST':
            train, test = self.mnist(root, transform)
        else:
            train, test = self.svhn(root, transform)
        return train, test, root

    def cifar10(self, root, transform):
        train = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform)

        return train, test

    def cifar100(self, root, transform):
        train = torchvision.datasets.CIFAR100(
            root=root, train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform)

        return train, test

    def mnist(self, root, transform):
        train = torchvision.datasets.MNIST(
            root=root, train=True, download=True, transform=transform)
        test = torchvision.datasets.MNIST(
            root=root, train=False, download=True, transform=transform)

        return train, test

    def svhn(self, root, transform):
        train = torchvision.datasets.SVHN(
            root=root, split='train', download=True, transform=transform)
        test = torchvision.datasets.SVHN(
            root=root, split='test', download=True, transform=transform)

        return train, test

    def global_contrast_normalization(self, image_dataset, lmbd, epsilon, s):
        image = []
        labels = []
        for each in image_dataset:
            img_temp = each[0].numpy()

            image.append(img_temp.transpose([1, 2, 0]))
            labels.append(each[1])
        image_contrast = []
        for each in image:
            image_mean = np.mean(each)
            image_data = each - image_mean

            contrast = np.sqrt(lmbd + np.mean(image_data ** 2))
            image_data = s * image_data / max(contrast, epsilon)
            image_contrast.append(image_data)
        image_contrast = np.array(image_contrast)
        return image_contrast, np.array(labels)

    def ZCA_whitening(self, dataset_name, image):

        image = image.reshape(image.shape[0], image.shape[1] * image.shape[2] * image.shape[3])
        image_norm = image / 255
        image_norm = image_norm - image_norm.mean(axis=0)
        image_cov = np.cov(image_norm, rowvar=False)
        U, S, V = np.linalg.svd(image_cov)
        epsilon = 0.1
        image_zca = U.dot(np.diag(1.0 / np.sqrt(S + epsilon))).dot(U.T).dot(image_norm.T).T
        image_zca_rescaled = (image_zca - image_zca.min()) / (image_zca.max() - image_zca.min())
        image_size = [28, 28, 1] if 'CIFAR' in dataset_name else [32, 32, 3]
        image_zca_rescaled = image_zca_rescaled.reshape(image.shape[0], image_size[0], image_size[1],
                                                        image_size[2])  # 28,28,1 for cifar
        return image_zca_rescaled

    def read_and_save(self, dataset_name, cifar_type=None):

        train_dataset, test_dataset, root = self.load_datasets(dataset_name, cifar_type)
        ds_name = os.path.join(self.out_dir, f'{dataset_name}.npy')

        if f'{dataset_name}.npy' not in os.listdir(self.out_dir):
            train_cont_set, train_labels = self.global_contrast_normalization(train_dataset, 10, 0.0001, 1)
            test_cont_set, test_labels = self.global_contrast_normalization(test_dataset, 10, 0.0001, 1)
            train_zca = self.ZCA_whitening(dataset_name, train_cont_set)
            test_zca = self.ZCA_whitening(dataset_name, test_cont_set)

            train = [list(each.transpose([2, 0, 1])) for each in train_zca]
            # train_set = np.array(train)
            test = [list(each.transpose([2, 0, 1])) for each in test_zca]
            # test_set = np.array(test)
            dataset = {'train': {'data': list(train), 'labels': list(train_labels)},
                       'test': {'data': list(test), 'labels': list(test_labels)}}
            # with open(ds_name, 'w') as out_file:
            #     json.dump(dataset, out_file)
            # np.save(ds_name, dataset)
        # with open(ds_name, 'rb') as saved_file:
        #     dataset = json.load(saved_file)
        dataset = np.load(ds_name, allow_pickle=True)

        print(type(dataset.item()))
        dataset = dataset.item()
        return dataset

    def get_loaders(self, dataset_name, valid_ratio, random_seed=10, shuffle=True, cifar_type=None):
        dataset_dict = self.read_and_save(dataset_name, cifar_type=cifar_type)

        train_zca = torch.Tensor(dataset_dict['train']['data'])
        test_zca = torch.Tensor(dataset_dict['test']['data'])
        train_labels = torch.LongTensor(dataset_dict['train']['labels'])
        test_labels = torch.LongTensor(dataset_dict['test']['labels'])

        train = TensorDataset(train_zca, train_labels)
        test = TensorDataset(test_zca, test_labels)
        validation = TensorDataset(train_zca, train_labels)

        len_train = len(train)
        indices = list(range(len_train))
        split = int(np.floor(valid_ratio * len_train))
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, validation_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        validation_sampler = SubsetRandomSampler(validation_idx)
        trainloader = torch.utils.data.DataLoader(
            train, batch_size=self.batch_size, sampler=train_sampler, num_workers=0)

        validationloader = torch.utils.data.DataLoader(
            validation, batch_size=self.batch_size, sampler=validation_sampler, num_workers=0)

        testloader = torch.utils.data.DataLoader(
            test, batch_size=self.batch_size, shuffle=False, num_workers=0)

        return trainloader, validationloader, testloader
