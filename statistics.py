import os
from utilities import *
from process_data import ProcessData
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
from runner import RunModel
from sklearn.metrics import confusion_matrix

class Statistics:
    def __init__(self, config_parameters):
        self.configuration = self.set_configuration(config_parameters)

    def set_configuration(self, parameters):
        dataset_name = parameters['dataset_name']
        environment = os.path.join('train_results', f'{dataset_name}_experiment_{parameters[f"{dataset_name}_exp_num"]}')
        split_info = 'original' if parameters['split'] else 'split'
        main_ds_folder = os.path.join(f'datasets/processed/{dataset_name}', split_info)
        check_dir(main_ds_folder)
        examples = os.path.join(main_ds_folder, 'examples')
        check_dir(examples)
        configuration = parameters

        configuration['split_info'] = split_info
        configuration['environment'] = environment
        configuration['main_folder'] = main_ds_folder
        configuration['examples'] = examples
        return configuration

    def get_distribution(self):
        types = ['train', 'test', 'dev'] if self.configuration['split'] else ['train', 'test']
        for each_type in types:
            process = ProcessData(self.configuration, each_type)
            label_distribution = Counter(process['label'])
            print(label_distribution)
            title = f"Data distribution for labels in {each_type} dataset of " \
                    f"{self.configuration['dataset_name']} model"
            plt.title(title)
            plt.xticks(np.arange(len(label_distribution.keys())))
            plt.bar(label_distribution.keys(), label_distribution.values())
            plt.plot()
            print(self.configuration['main_folder'])
            print(f'{self.configuration["dataset_name"]}_{each_type}_dist.png')
            figure_path = os.path.join(self.configuration['main_folder'],
                                       f'{self.configuration["dataset_name"]}_{each_type}_dist.png')
            print(figure_path)
            plt.savefig(figure_path)
            plt.close()

    def plot_results(self, is_accuracy: bool = True) -> None:
        """
        Method is used to plot accuracy/loss graphs after training session is over, according to provided variable
        :param is_accuracy: boolean variable specifies the type of data will be plotted
        :return: None
        """
        results_file = os.path.join(self.configuration['environment'], 'results.pickle')
        with open(results_file, 'rb') as result_data:
            result_dict = pickle.load(result_data)
        metric_key = 'acc' if is_accuracy else 'loss'
        dev_data = list()
        train_data = list()

        ordered = OrderedDict(sorted(result_dict.items()))

        for epoch, results in ordered.items():
            dev_data.append(results[f'dev_{metric_key}'])
            train_data.append(results[f'train_{metric_key}'])
        plt.figure()
        plt.title(f'{metric_key.title()} results over {len(result_dict.keys())} epochs for {self.configuration["dataset_name"].upper()}')
        plt.plot(list(result_dict.keys()), train_data, 'g', label='Train')
        plt.plot(list(result_dict.keys()), dev_data, 'r', label='Validation')
        plt.grid()
        plt.xlabel('Number of epochs')
        plt.ylabel(f'{metric_key.title()} results')
        plt.legend(loc=4)
        figure_path = os.path.join(self.configuration['environment'], f'{metric_key}_plot.png')
        plt.savefig(figure_path)
        plt.show()

    def get_confusion_matrix(self) -> None:
        """
        Method is used to generate confusion matrix and call specific method to visualize it
        :return: None
        """
        runner = RunModel(self.configuration)
        best_epoch = runner.get_epoch(is_best=True)
        file_name = os.path.join(self.configuration['environment'], f'inferences/inferences_{best_epoch}.pickle')
        with open(file_name, 'rb') as inference_dict:
            inference_data = pickle.load(inference_dict)

        conf_matrix = confusion_matrix(inference_data['target'], inference_data['prediction'])
        self.plot_confusion_matrix(conf_matrix, set(inference_data['target']))

    def plot_confusion_matrix(self, conf_matrix: np.array, labels) -> None:
        """
        Method is used for visualization of provided confusion matrix
        :param conf_matrix: numpy array which expresses confusion matrix
        :param cascade: boolean variable which specifies confusion matrix is generated for cascade operation or not
        :return: None
        """

        plt.figure(figsize=(8, 6), dpi=100)
        sns.set(font_scale=1.1)

        ax = sns.heatmap(conf_matrix, annot=True, fmt='d', )

        ax.set_xlabel("Predicted Labels", fontsize=14, labelpad=20)
        ax.xaxis.set_ticklabels(labels)

        ax.set_ylabel("Actual Labels", fontsize=14, labelpad=20)
        ax.yaxis.set_ticklabels(labels)
        ax.set_title(f"Confusion Matrix for {self.configuration['dataset_name']}", fontsize=14, pad=20)
        image_name = os.path.join(self.configuration['environment'], f'confusion_matrix.png')
        plt.savefig(image_name)
        plt.show()

    def provide_stats(self, before=True):
        if before:
            self.get_distribution()
        else:
            self.plot_results(is_accuracy=True)
            self.plot_results(is_accuracy=False)
            if self.configuration['dataset_name'] != 'cifar100':
                self.get_confusion_matrix()
