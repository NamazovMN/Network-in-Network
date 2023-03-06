import os
import pickle

import numpy as np
from tqdm import tqdm

from collect import CollectDataset
from utilities import check_dir


class ProcessData:
    """
    Method is utilized to process provided dataset according to the provided arguments
    """
    def __init__(self, config_parameters: dict, ds_type: str):
        """
        Method is utilized as an initializer of the class
        :param config_parameters: all required parameters for the project
        :param ds_type: dataset type that will be processed and collected by the class
        """
        self.ds_type = ds_type
        self.configuration = self.set_configuration(config_parameters)
        self.dataset = self.process_data()

    def set_configuration(self, parameters: dict):
        """
        Method is utilized to extract required parameters for the sub-task from all required parameters of the project
        :param parameters: all required parameters for the project
        :return: sub-task relevant parameters
        """
        output_dir = parameters['input_dir']
        check_dir(output_dir)
        split_info = 'split' if parameters['split'] else 'original'
        dataset_dir = os.path.join(output_dir, f"processed/{parameters['dataset_name']}/{split_info}")

        check_dir(dataset_dir)
        return {
            'parameters': parameters['split'],
            'ds_name': parameters['dataset_name'],
            'out_dir': self.set_output_dir(parameters, dataset_dir),
            'source_set': self.get_dataset(parameters),
            'lambda': parameters['lambda'],
            'epsilon': parameters['epsilon'],
            's': parameters['s'],
            'global_contrast': parameters['global_contrast'],
            'zca_white': parameters['zca_white']
        }

    @staticmethod
    def set_output_dir(parameters: dict, ds_dir: str) -> str:
        """
        Method is utilized to generate output directory
        :param parameters: all required parameters for the project
        :param ds_dir: main dataset directory which was set according to the split information
        :return: output directory according to the provided arguments (zca, gc)
        """
        ds_info = 'dataset'
        gc_info = '_gc' if parameters['global_contrast'] else ''
        zca_info = '_zca' if parameters['zca_white'] else ''
        out_folder = os.path.join(ds_dir, f"{ds_info}{gc_info}{zca_info}")
        check_dir(out_folder)
        return out_folder

    def get_dataset(self, parameters: dict) -> dict:
        """
        Method is utilized to collect specific dataset to be processed
        :param parameters: all required parameters for the project
        :return:dataset dictionary according to the specific dataset type
        """
        ds_collect = CollectDataset(parameters)
        return ds_collect[self.ds_type]

    def global_contrast_normalization(self) -> dict:
        """
        Method is utilized to create the dataset which was globally contrast normalized
        :return: gcn dataset
        """

        result_dataset = {
            'data': list(),
            'label': self.configuration['source_set']['label']
        }
        ti = tqdm(self.configuration['source_set']['data'], total=len(self.configuration['source_set']['data']),
                  desc=f'Global contrast normalization is applied for {self.ds_type} of {self.configuration["ds_name"]}'
                  )
        for image in ti:
            image_data = image - np.mean(image)
            contrast = np.sqrt(self.configuration['lambda'] + np.mean(image_data ** 2))

            image_result = self.configuration['s'] * image_data / max(contrast, self.configuration['epsilon'])
            result_dataset['data'].append(image_result)
        result_dataset['data'] = np.array(result_dataset['data'])
        return result_dataset

    def compute_zca_matrix(self, image: np.array) -> np.array:
        """
        Method is utilized to generate ZCA matrix for the specific image data
        :param image: specific image from the input dataset
        :return: ZCA matrix for transforming image
        """
        sigma = np.cov(image, rowvar=True)
        u, s, v = np.linalg.svd(sigma)
        return np.dot(u, np.dot(np.diag(1.0 / np.sqrt(s + self.configuration['epsilon'], u.T))))

    def zca_whitening(self, dataset: dict) -> dict:
        """
        Method is utilized to apply ZCA whitening to the dataset
        :param dataset: provided input dataset
        :return: ZCA whitened dataset dictionary
        """
        print('ZCA whitening is applied')
        ds_array = dataset['data']
        dataset_flattened = ds_array.reshape((ds_array.shape[0], ds_array.shape[1] * ds_array.shape[2] * ds_array.shape[3]))
        normalized_ds = dataset_flattened / 255.
        normalized_ds = normalized_ds - normalized_ds.mean(axis=0)

        covariance = np.cov(normalized_ds, rowvar=False)
        u, s, v = np.linalg.svd(covariance)
        zca_dataset = u.dot(np.diag(1.0 / np.sqrt(s + self.configuration['epsilon']))).dot(u.T).dot(normalized_ds.T).T
        result_dataset = (zca_dataset - zca_dataset.min()) / (zca_dataset.max() - zca_dataset.min())

        return {
            'data': result_dataset.reshape(ds_array.shape[0], ds_array.shape[1], ds_array.shape[2], ds_array.shape[3]),
            'label': dataset['label']
        }

    def process_data(self) -> dict:
        """
        Method is utilized as the main function of the class which performs all consecutive processes
        :return: resulting dataset (dictionary)
        """
        file_name = os.path.join(self.configuration['out_dir'], f"{self.ds_type}.pickle")
        if not os.path.exists(file_name):

            dataset = self.global_contrast_normalization() if self.configuration['global_contrast'] \
                else self.configuration['source_set']
            dataset = self.zca_whitening(dataset) if self.configuration['zca_white'] else dataset
            with open(file_name, 'wb') as ds_data:
                pickle.dump(dataset, ds_data)
        with open(file_name, 'rb') as ds_data:
            dataset = pickle.load(ds_data)

        return dataset

    def __getitem__(self, item: str) -> np.array:
        """
        Method is utilized to collect specific part of the given dataset according to the provided dataset type
        :param item: string that specifies whether data or label list is required
        :return: list of requested information (data or labels)
        """
        return self.dataset[item]

    def __len__(self) -> int:
        """
        Method is utilized to get length of the specific dataset
        :return: number of data in the given set
        """
        return len(self.dataset['data'])
