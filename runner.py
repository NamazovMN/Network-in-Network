import pickle
from typing import Any

from tqdm import tqdm

from utilities import *
from torch.utils.data import DataLoader
from dataset import Datasets
from model import NetworkInNetwork
from torch import nn
from torch.optim import Adam, SGD
from sklearn.metrics import f1_score


class RunModel:
    def __init__(self, config_parameters):
        self.configuration = self.set_configuration(config_parameters)
        self.model = self.set_model(config_parameters, len(self.configuration['labels']),
                                    self.configuration['input_channels'])
        self.save_architecture()
        self.loss_fn = self.set_loss_fn()
        self.optimizer = self.set_optimizer()

    def set_configuration(self, parameters):
        train_results = 'train_results'
        check_dir(train_results)
        experiment_info = f"{parameters['dataset_name']}_exp_num"
        experiment_environment = os.path.join(train_results,
                                              f"{parameters['dataset_name']}_experiment_{parameters[experiment_info]}")
        check_dir(experiment_environment)
        checkpoints_dir = os.path.join(experiment_environment, 'ckpt')
        check_dir(checkpoints_dir)

        types = ['train', 'test', 'dev'] if parameters['split'] else ['train', 'test']

        dataloaders = {each_type: self.get_loaders(parameters, each_type) for each_type in types}
        init_channels, labels = self.get_labels(parameters)
        model_name = parameters['dataset_name']
        inference_dir = os.path.join(experiment_environment, 'inferences')
        check_dir(inference_dir)
        return {
            'optimizer': parameters['optimizer'],
            'learning_rate': parameters[f'{model_name}_learning_rate'],
            'model_name': model_name,
            'input_channels': init_channels,
            'labels': labels,
            'epochs': parameters['epochs'],
            'environment': experiment_environment,
            'ckpt': checkpoints_dir,
            'batch_size': parameters['batch_size'],
            'loaders': dataloaders,
            'device': parameters['device'],
            'resume_training': parameters['resume_training'],
            'inference_dir': inference_dir
        }

    @staticmethod
    def set_loss_fn() -> nn.CrossEntropyLoss:
        """
        Method is used to set loss function according to the provided parameters
        :return: Loss function
        """
        return nn.CrossEntropyLoss()

    def set_model(self, parameters: dict, num_labels: int, input_channels: int) -> NetworkInNetwork:
        """

        :param parameters: All required parameters for the project
        :param num_labels: number of labels for given dataset
        :param input_channels: number of channels in one image
        :return: Network In Network model
        """
        return NetworkInNetwork(config_parameters=parameters, num_labels=num_labels, input_channels=input_channels).to(
            self.configuration['device'])

    def set_optimizer(self) -> Any:
        """
        Method is used to set optimizer according to the provided parameters
        :return: Optimizer for the model
        """
        if self.configuration['optimizer'] == 'Adam':
            optimizer = Adam(params=self.model.parameters(), lr=self.configuration['learning_rate'])
        elif self.configuration['optimizer'] == 'SGD':
            optimizer = SGD(params=self.model.parameters(), lr=self.configuration['learning_rate'], momentum=0.8)
        else:
            raise Exception('There is not such optimizer in our scenarios. You should choose one of SGD or Adam')
        return optimizer

    @staticmethod
    def get_labels(parameters: dict) -> tuple:
        """
        Method is utilized to extract required data from the dataset
        :param parameters: all required parameters for the project
        :return: tuple which contains number of channels and number of labels
        """
        dataset = Datasets(parameters, 'test')
        return dataset.data.shape[1], dataset.set_labels

    @staticmethod
    def get_loaders(parameters: dict, ds_type: str) -> DataLoader:
        """
        Method is utilized to generate specific data loader for the given dataset
        :param parameters: all required parameters for the project
        :param ds_type: specifies type of the dataset
        :return: specific data loader object
        """
        return DataLoader(Datasets(parameters, ds_type), shuffle=True, batch_size=parameters['batch_size'])

    def run_step(self, batch: dict, train: bool = True) -> tuple:
        """
        Method is utilized to perform step calculations
        :param batch: batch information contains data and labels for specific batch
        :param train: boolean variable specifies whether step belongs to training or evaluation
        :return: tuple of step loss, step accuracy, list of predictions and targets per step
        """
        data = batch['data'].to(self.configuration['device'])
        labels = batch['label'].to(self.configuration['device'])
        if train:
            self.optimizer.zero_grad()
        output = self.model(data.to(self.configuration['device']))

        loss = self.loss_fn(output, labels)

        if train:
            loss.backward()
            self.optimizer.step()

        prediction, targets, accuracy = self.compute_accuracy(output, labels)
        return loss.item(), accuracy, prediction, targets

    @staticmethod
    def compute_accuracy(prediction: torch.Tensor, labels: torch.Tensor) -> tuple:
        """
        Method is utilized to compute accuracy according to the given predictions and targets
        :param prediction: tensor of the predictions
        :param labels: tensor of the targets
        :return: tuple of list of predictions, targets and accuracy for epoch
        """
        predicted_labels = prediction.argmax(-1).tolist()
        targets = labels.tolist()
        accuracy = sum([p == t for p, t in zip(predicted_labels, targets)])
        return predicted_labels, targets, accuracy

    def set_epoch_range(self) -> range:
        """
        Method is utilized to set epoch range according to the resume training information
        :return: range of epochs between choice and the number of epochs
        """
        init = -1
        if self.configuration['resume_training']:
            init = self.get_epoch(is_best=False)
        return range(init + 1, self.configuration['epochs'])

    def train_epoch(self) -> None:
        """
        Method is utilized to train the model for specific number of epochs
        :return:
        """
        epoch_range = self.set_epoch_range()
        for epoch in epoch_range:
            self.model.train()
            num_data = 0
            epoch_loss = 0
            epoch_accuracy = 0
            num_batches = len(self.configuration['loaders']['train'])
            ti = tqdm(iterable=self.configuration['loaders']['train'],
                      total=num_batches, desc='Epoch 0: Training =>', leave=True)

            for batch in ti:
                step_loss, step_accuracy, prediction, targets = self.run_step(batch)
                epoch_loss += step_loss
                epoch_accuracy += step_accuracy
                num_data += len(prediction)
                ti.set_description(f'Epoch {epoch} Training => '
                                   f'Train Loss: {epoch_loss / num_batches: .4f} '
                                   f'Train Accuracy: {epoch_accuracy / num_data: .4f}')
            eval_loss, eval_acc, eval_f1 = self.evaluate_epoch(epoch)
            result_dict = {
                'train_loss': epoch_loss / num_batches,
                'dev_loss': eval_loss,
                'train_acc': epoch_accuracy / num_data,
                'dev_acc': eval_acc,
                'dev_f1': eval_f1
            }
            self.save_parameters(result_dict, epoch)

    def save_architecture(self) -> None:
        """
        Method is utilized to save the model architecture
        :return:
        """
        architecture = os.path.join(self.configuration['environment'], 'architecture.pickle')
        with open(architecture, 'wb') as architecture_data:
            pickle.dump(self.model.model_architecture, architecture_data)

    def save_parameters(self, epoch_dict: dict, epoch: int) -> None:
        """
        Method is utilized to save model parameters and training results after each epoch
        :param epoch_dict: dictionary that contains numerical results after each epoch
        :param epoch: specific epoch information
        :return: None
        """
        results_file = os.path.join(self.configuration['environment'], 'results.pickle')
        if not os.path.exists(results_file):
            result_dict = {epoch: epoch_dict}
        else:
            with open(results_file, 'rb') as data:
                result_dict = pickle.load(data)
            result_dict[epoch] = epoch_dict
        with open(results_file, 'wb') as data:
            pickle.dump(result_dict, data)
        print(f'Numerical results of the {self.configuration["model_name"]} were saved successfully!')

        model_dict_name = os.path.join(self.configuration['ckpt'],
                                       f"epoch_{epoch}_dev_loss_{epoch_dict['dev_loss']: .3f}"
                                       f"_train_loss_{epoch_dict['train_loss']: .3f}"
                                       f"_f1_{epoch_dict['dev_f1']: .3f}")
        optimizer_dict_name = os.path.join(self.configuration['ckpt'], f'epoch_{epoch}_optim')
        torch.save(self.model.state_dict(), model_dict_name)
        torch.save(self.optimizer.state_dict(), optimizer_dict_name)
        print(f'Model and optimizer parameters for epoch {epoch} were saved successfully!')
        print(f'{20 * "<"}{20 * ">"}')

    def load_model(self, is_best: bool = False) -> None:
        """
        Method is utilized to load the model according to the provided variable
        :param is_best: boolean variable to specify whether the best epoch will be loaded or the last one
        :return: None
        """
        epoch = self.get_epoch(is_best)
        checkpoints = {'model_ckpt': str, 'optim_ckpt': f"epoch_{epoch}_optim"}
        for each_file in os.listdir(self.configuration["ckpt"]):
            if f"epoch_{epoch}_dev" in each_file:
                checkpoints['model_ckpt'] = each_file
                break
        if not checkpoints['model_ckpt']:
            raise NotImplemented('Model was not trained that much epochs! Check your files and train the model!')
        for ckpt_type, ckpt in checkpoints.items():
            checkpoints[ckpt_type] = os.path.join(self.configuration['ckpt'], ckpt)
        self.model.load_state_dict(torch.load(checkpoints['model_ckpt'], map_location=self.configuration['device']))
        self.optimizer.load_state_dict(
            torch.load(torch.load(checkpoints['optim_ckpt']), map_location=self.configuration['device']))
        self.model.eval()

    def get_epoch(self, is_best: bool = False, metric: str = 'dev_f1') -> int:
        """
        Method is utilized to get specific epoch according to the requirements
        :param is_best: boolean variable to specify whether the best epoch will be loaded or the last one
        :param metric: metric value that the best model will be loaded according to it
        :return: chosen epoch information
        """
        result_file = os.path.join(self.configuration['environment'], 'results.pickle')
        if not os.path.exists(result_file):
            raise NotImplemented('Model was not trained! You need to train the model first!')
        with open(result_file, 'rb') as data:
            result_dict = pickle.load(data)

        if is_best:
            check_data = {epoch: value[metric] for epoch, value in result_dict.items()}
            chosen = min(check_data, key=check_data.get) if 'loss' in metric else max(check_data, key=check_data.get)
        else:
            chosen = max(result_dict.keys())
        return chosen

    def evaluate_epoch(self, epoch: int):
        """
        Method is utilized to evaluate for the specific epoch
        :param epoch: provided epoch value
        :return: tuple of average development loss and accuracy, and f1 score for the epoch
        """
        self.model.eval()
        loader_choice = 'dev' if len(self.configuration['loaders']) == 3 else 'test'
        num_batches = len(self.configuration['loaders'][loader_choice])
        ti = tqdm(iterable=self.configuration['loaders'][loader_choice], total=num_batches,
                  desc=f'Epoch {epoch}: Evaluation => ')
        inference_dict = {'targets': list(), 'predictions': list()}
        eval_loss = 0
        eval_acc = 0
        step_size = 0
        for batch in ti:
            step_loss, step_accuracy, prediction, targets = self.run_step(batch, train=False)
            step_size += len(prediction)
            eval_loss += step_loss
            eval_acc += step_accuracy
            inference_dict['targets'].extend(targets)
            inference_dict['predictions'].extend(prediction)
            ti.set_description(f'Epoch {epoch}: Evaluation => Loss: {eval_loss / num_batches: .4f}, '
                               f'Accuracy: {eval_acc / step_size: .4f}')
        f1 = f1_score(inference_dict['targets'], inference_dict['predictions'], average='macro')
        print(f'Epoch {epoch}: F1: {f1: .4f}')
        inference_dir = os.path.join(self.configuration['inference_dir'], f'inference_{epoch}.pickle')
        with open(inference_dir, 'wb') as inference_data:
            pickle.dump(inference_dict, inference_data)
        return eval_loss / num_batches, eval_acc / step_size, f1
