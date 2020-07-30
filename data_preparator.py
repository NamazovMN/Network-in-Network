import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader
import os
from numpy import save, load
class DataPreparator(object):
    def __init__(self, dataset_name, batch_size):
        self.dataset_name = dataset_name
        self.batch_size = batch_size

    def load_datasets(self):
        if self.dataset_name == "cifar_10":
            root = "./data/CIFAR10"
            print("Dataset is CIFAR10")
            transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
            )
            trainset = torchvision.datasets.CIFAR10(
                root=root, train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10(
                root=root, train=False, download=True, transform=transform)

            validationset = torchvision.datasets.CIFAR10(
                root=root, train=True, download=True, transform=transform)
        elif self.dataset_name == "cifar_100":
            print("Dataset is CIFAR100")

            root = "./data/CIFAR100"
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
            trainset = torchvision.datasets.CIFAR100(
                root=root, train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR100(
                root=root, train=False, download=True, transform=transform)

            validationset = torchvision.datasets.CIFAR100(
                root=root, train=True, download=True, transform=transform)
        elif self.dataset_name == "svhn":
            print("Dataset is SVHN")

            root = "./data/SVHN"
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
            trainset = torchvision.datasets.SVHN(
                root=root, split="train", download=True, transform=transform)
            testset = torchvision.datasets.SVHN(
                root=root, split='test', download=True, transform=transform)

            validationset = torchvision.datasets.SVHN(
                root=root, split = 'train', download=True, transform=transform)        
        elif self.dataset_name == "mnist":
            print("Dataset is MNIST")

            root = "./data/MNIST"
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5),(0.5))])
            trainset = torchvision.datasets.MNIST(
                root=root, train=True, download=True, transform=transform)
            testset = torchvision.datasets.MNIST(
                root=root, train=False, download=True, transform=transform)

            validationset = torchvision.datasets.MNIST(
                root=root, train=True, download=True, transform=transform)
        else:
            print("there is no such root")

        

        return trainset, testset, root

    def global_contrast_normalization(self, image_dataset, lmbd, epsilon, s):
        image = []
        labels = []
        for each in image_dataset:
            img_temp = each[0].numpy()

            image.append(img_temp.transpose([1,2,0]))
            labels.append(each[1])
        image_contrast = []
        for each in image:
            image_mean = np.mean(each)
            image_data = each-image_mean

            contrast = np.sqrt(lmbd + np.mean(image_data**2))
            image_data = s*image_data/max(contrast, epsilon)
            image_contrast.append(image_data)
        image_contrast = np.array(image_contrast)
        return image_contrast, np.array(labels)
    
    def ZCA_whitening(self, image):

        image = image.reshape(image.shape[0],image.shape[1]*image.shape[2]*image.shape[3])
        image_norm = image/255
        image_norm = image_norm-image_norm.mean(axis = 0)
        image_cov = np.cov(image_norm, rowvar = False)
        U,S,V = np.linalg.svd(image_cov)
        epsilon = 0.1
        image_ZCA = U.dot(np.diag(1.0/np.sqrt(S+epsilon))).dot(U.T).dot(image_norm.T).T
        image_ZCA_rescaled = (image_ZCA-image_ZCA.min())/(image_ZCA.max()-image_ZCA.min())
        image_ZCA_rescaled = image_ZCA_rescaled.reshape(image.shape[0], 32,32,3) #28,28,1 for cifar
        return image_ZCA_rescaled
    
    def prepare_dataloaders(self, random_seed = 10, valid_ratio = 0.2, shuffle = True):
        train_set = []
        test_set = []
        train_labels = []
        test_labels = []
        train_dataset, test_dataset, root = self.load_datasets()
        if not os.path.exists(root+"/generated_data"):
            os.makedirs(root+"/generated_data")
            train_cont_set, train_labels = self.global_contrast_normalization(train_dataset, 10,0.0001,1)
            test_cont_set, test_labels = self.global_contrast_normalization(test_dataset, 10,0.0001,1)
            train_ZCA = self.ZCA_whitening(train_cont_set)
            test_ZCA = self.ZCA_whitening(test_cont_set)
            
            train = [each.transpose([2,0,1]) for each in train_ZCA]
            train_set = np.array(train)
            test = [each.transpose([2,0,1]) for each in test_ZCA]
            test_set = np.array(test)
            save( root+"/generated_data/tr_data.npy",train_set)
            save(root+"/generated_data/test_data.npy", test_set)
            save( root+"/generated_data/tr_labels.npy",train_labels)
            save( root+"/generated_data/test_labels.npy",test_labels)

        else:
            train_set = load(root+"/generated_data/tr_data.npy")
            test_set = load(root+"/generated_data/test_data.npy")
            train_labels = load(root+"/generated_data/tr_labels.npy")
            test_labels = load(root+"/generated_data/test_labels.npy")

        train_ZCA = torch.Tensor(train_set)
        test_ZCA = torch.Tensor(test_set)
        train_labels = torch.LongTensor(train_labels)
        test_labels = torch.LongTensor(test_labels)

        trainset = TensorDataset(train_ZCA, train_labels)
        testset = TensorDataset(test_ZCA, test_labels)
        validationset = TensorDataset(train_ZCA, train_labels)

       
        
        len_train = len(trainset)
        indices = list(range(len_train))
        split = int(np.floor(valid_ratio*len_train))
        if(shuffle):
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        
        train_idx, validation_idx = indices[split:], indices[:split]
        trainSampler = SubsetRandomSampler(train_idx)
        validationSampler = SubsetRandomSampler(validation_idx)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, sampler = trainSampler, num_workers=0)

        validationloader = torch.utils.data.DataLoader(
            validationset, batch_size = self.batch_size, sampler = validationSampler, num_workers = 0)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        return trainloader, validationloader, testloader
