from typing import Any

import torch
import torch.nn as nn
from torch.nn import Conv2d, AvgPool2d, MaxPool2d

class NIN_Architecture:
    """
    Class is utilized to build model architecture dynamically
    """
    def __init__(self, config_parameters: dict, num_classes: int, input_channels: int):
        """
        Method is utilized as an initializer of the class
        :param config_parameters: all required parameters for the project
        :param num_classes: number of labels
        :param input_channels: number of input image's channels
        """
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.configuration = self.set_configuration(config_parameters)
        self.compatibility_check()
        self.pools = self.pool_choices()
        self.model_arch = self.set_model_architecture()

    def set_configuration(self, parameters: dict) -> dict:
        """
        Method is utilized to extract required parameters for the sub-task from the all parameters of the project
        :param parameters: all required parameters for the project
        :return: required parameters for the specific sub-task
        """
        model_name = parameters['dataset_name']

        return {
            'block_out': parameters[f'{model_name}_block_out'],
            'block_kernels': parameters[f'{model_name}_block_kernels'],
            'block_strides': parameters[f'{model_name}_block_strides'],
            'block_paddings': parameters[f'{model_name}_block_paddings'],
            'dropout': parameters[f'{model_name}_dropouts'],
            'pools': parameters[f'{model_name}_pool_choice'],
            'pool_kernels': parameters[f'{model_name}_pool_kernels'],
            'pool_strides': parameters[f'{model_name}_pool_strides'],
            'pool_paddings': parameters[f'{model_name}_pool_paddings'],
            'batch_norms': parameters[f'{model_name}_batch_norms']
        }

    def compatibility_check(self) -> None:
        """
        Method is utilized to check block compatibilities
        :return: None
        """
        default = len(self.configuration['block_out'])
        for parameter,each in self.configuration.items():
            if len(each) != default:
                raise IndexError(f'Number of blocks in {parameter} is not compatible with default '
                                 f'number of blocks : {default}')
            if 'block' in parameter and 'out' not in parameter:
                self.check_deep_compatibility(each)

    def check_deep_compatibility(self, block_info: list) -> None:
        """
        Method is utilized to check compatibility in lower levels for blocks
        :param block_info: list of block outputs
        :return: None
        """
        for idx, each in enumerate(block_info):
            if len(each) != len(self.configuration['block_out'][idx]):
                print(len(each))
                raise IndexError(f'Number of data is not compatible with block size of {idx}. block')


    def set_block(self, block_idx: int, input_channels: int) -> dict:
        """
        Method is utilized to generate specific block according to the provided information
        :param block_idx: index of the block
        :param input_channels: number of channels
        :return: dictionary that contains all corresponding layers in the requested block
        """
        block = dict()
        init = input_channels
        for idx, output in enumerate(self.configuration['block_out'][block_idx]):
            block[f'cnn_{idx}'] = nn.Conv2d(
                in_channels=init,
                out_channels=output,
                kernel_size=self.configuration['block_kernels'][block_idx][idx],
                stride=self.configuration['block_strides'][block_idx][idx],
                padding=self.configuration['block_paddings'][block_idx][idx]
            )
            block[f'activation_{idx}'] = nn.ReLU()
            init = output
        if self.configuration['batch_norms']:
            block[f'batch_norm'] = nn.BatchNorm2d(num_features=self.configuration['block_out'][block_idx][-1])
        block['pooling'] = self.set_pool(block_idx)
        block['dropout'] = nn.Dropout(self.configuration['dropout'][block_idx])
        return block

    def pool_choices(self) -> dict:
        """
        Method is utilized to make pooling layer choice automatic
        :return: dictionary of pooling layers
        """
        pool_layers = {
            'avg': AvgPool2d,
            'max': MaxPool2d
        }
        return pool_layers

    def set_pool(self, block_idx: int) -> Any:
        """
        Method is utilized to generate pooling layer for the requested block
        :param block_idx: index of the requested block
        :return: pooling layer for the block
        """
        pool_choice = 'avg' if self.configuration['pools'][block_idx] else 'max'
        return self.pools[pool_choice](
            kernel_size=self.configuration['pool_kernels'][block_idx],
            stride=self.configuration['pool_strides'][block_idx],
            padding=self.configuration['pool_paddings'][block_idx]
        )

    def set_model_architecture(self) -> dict:
        """
        Method is utilized to build model architecture
        :return: dictionary that contains all required layers for the model
        """
        model_arch = dict()
        for block_idx, block in enumerate(self.configuration['block_out']):
            input_channels = self.input_channels if block_idx == 0 else self.configuration['block_out'][block_idx-1][-1]
            block = self.set_block(block_idx, input_channels)
            for layer_name, layer in block.items():
                model_arch[f"block_{block_idx}_{layer_name}"] = layer
        return model_arch

    def __iter__(self):
        """
        Method is utilized as a generator for iterating over model layers
        :return: yields tuple of layer name and layer itself
        """
        for layer_name, layer in self.model_arch.items():
            yield layer_name, layer


class NetworkInNetwork(nn.Module):
    """
    Class is a model object
    """
    def __init__(self, config_parameters: dict, num_labels: int, input_channels: int):
        """
        Method is utilized as an initializer of the class
        :param config_parameters: all required parameters for the project
        :param num_labels: number of labels
        :param input_channels: number of input image's channels
        """
        super(NetworkInNetwork, self).__init__()
        self.num_labels = num_labels
        self.model_architecture = NIN_Architecture(config_parameters, num_classes=num_labels, input_channels=input_channels)

        for name, module in self.model_architecture:
            self.add_module(name, module)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Method is utilized to perform feedforward process of the model
        :param input_data: input tensor
        :return: output tensor
        """

        for module in self.children():
            input_data = module(input_data)
        output_data = input_data.view(input_data.size(0), self.num_labels)

        return output_data


