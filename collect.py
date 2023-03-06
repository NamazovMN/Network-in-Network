import os
import pickle
from typing import Any

import numpy as np
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN

from utilities import *


class CollectDataset:
    """
    Class is utilized to collect the all data types from the torchvision datasets and save them in one file
    """
    def __init__(self, config_parameters: dict):
        """
        Method is utilized as initializer of the class
        :param config_parameters: all required parameters for the project
        """
        self.configuration = self.set_configuration(config_parameters)
        self.collector = self.get_collector()
        self.datasets = self.process_data()

    def set_configuration(self, parameters: dict) -> dict:
        """
        Method extracts required parameters for this sub-task from all parameters
        :param parameters: all required parameters for the project
        :return: required parameters for the sub-task
        """
        source = os.path.join(parameters['input_dir'], 'source')
        check_dir(source)
        return {
            'output_dir': self.get_output_dir(parameters),
            'ds_name': parameters['dataset_name'],
            'split': parameters['split'],
            'raw_source': source
        }

    @staticmethod
    def get_output_dir(parameters: dir) -> str:
        """
        Method is utilized to generate output directory. Since the project has complicated shape, we generated it
        separately
        :param parameters: all required parameters for the project
        :return: main output folder according to the requirements (split information is considered)
        """
        processed_dir = os.path.join(parameters['input_dir'], 'processed')
        check_dir(processed_dir)
        dataset_folder = os.path.join(processed_dir, parameters['dataset_name'])
        check_dir(dataset_folder)
        subdir = os.path.join(dataset_folder, 'split' if parameters['split'] else 'original')
        check_dir(subdir)

        return subdir

    def get_collector(self) -> Any:
        """
        Method is utilized to make data collecting process automatic
        :return:
        """
        collector_dict = {
            'mnist': MNIST,
            'cifar10': CIFAR10,
            'cifar100': CIFAR100,
            'svhn': SVHN
        }
        return collector_dict[self.configuration['ds_name']]

    def collect_dataset(self) -> dict:
        """
        Method is utilized to collect dataset according to split info. If there is not any split percentage, then it
        will collect only train and test sets. Otherwise, it will split train set into training and development sets
        :return: dataset dictionary in which keys are dataset types and values are data dictionary
        """

        dataset = dict()
        for each_type in ['train', 'test']:
            raw_dataset = self.get_dataset(each_type)
            if each_type == 'train' and self.configuration['split']:
                generated_sets = self.split_set(raw_dataset)
                for key, ds in generated_sets.items():
                    dataset[key] = ds
            else:
                dataset[each_type] = self.process_original(raw_dataset)

        for key, ds in dataset.items():
            shape = ds['data'][0].shape
            data = np.array(ds['data']) if len(shape) == 3 \
                else np.array(ds['data']).reshape((len(ds['data']), shape[0], shape[1], 1))
            dataset[key] = {
                'data': data,
                'label': ds['label']
            }

        return dataset

    @staticmethod
    def process_original(dataset: Any) -> dict:
        """
        Method is utilized to process original dataset, which is collect PIL images and transform them into array
        :param dataset: raw dataset which was collected from the torchvision
        :return: dictionary of the processed datasets
        """
        processed_dataset = {'data': list(), 'label': list()}
        for (image, label) in dataset:
            processed_dataset['data'].append(np.array(image))
            processed_dataset['label'].append(label)
        return processed_dataset

    def split_set(self, raw_set: Any) -> dict:
        """
        Method is utilized to split the original train set into development and training sets if it is requested
        :param raw_set: original dataset from torchvision
        :return: dictionary of resulting sets
        """
        labels = [label for (_, label) in raw_set]
        dataset = {label: {'data': list(), 'label': list()} for label in set(labels)}
        for (image, label) in raw_set:
            dataset[label]['data'].append(np.array(image))
            dataset[label]['label'].append(label)
        datasets = {ds_type: {'data': list(), 'label': list()} for ds_type in ['train', 'dev']}
        for ds_type in ['train', 'dev']:
            for label, ds in dataset.items():
                separator = int(len(ds['data']) * self.configuration['split'])
                datasets[ds_type]['data'].extend(
                    ds['data'][0: separator] if ds_type == 'train' else ds['data'][separator::])
                datasets[ds_type]['label'].extend(
                    ds['label'][0: separator] if ds_type == 'train' else ds['label'][separator::])

        return datasets

    def get_dataset(self, ds_type: str) -> Any:
        """
        Method is utilized to collect the original dataset from the torchvision base
        :param ds_type: type of data which can be training or test (dev is generated by us, if it is requested)
        :return: raw dataset for requested type
        """
        if self.configuration['ds_name'] != 'svhn':
            train = True if ds_type == 'train' else False
            dataset = self.collector(root=self.configuration['raw_source'], train=train,
                                     download=True)

        else:
            train = ds_type
            dataset = self.collector(root=self.configuration['raw_source'], split=train,
                                     download=True)

        return dataset

    def process_data(self) -> dict:
        """
        Method is utilized to combine all consecutive processes in one. At the end it saves dataset in order to prevent
        redundant repetition of processes
        :return: resulting dataset in form of dictionary
        """
        filename = os.path.join(self.configuration['output_dir'], 'datasets.pickle')
        if not os.path.exists(filename):
            datasets = self.collect_dataset()
            with open(filename, 'wb') as ds_data:
                pickle.dump(datasets, ds_data)
        with open(filename, 'rb') as ds_data:
            datasets = pickle.load(ds_data)
        return datasets

    def __getitem__(self, item: str):
        """
        Method is utilized to get specific dataset according to the provided data type which can be training, test, dev
        :param item: dataset type which will be collected
        :return: selected dataset dictionary, where keys are data and label
        """
        if item not in self.datasets.keys():
            raise IndexError(f'There is not such dataset in process! It must be one of {list(self.datasets.keys())}')
        return self.datasets[item]
