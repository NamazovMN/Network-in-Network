import torch
from torch.utils.data import Dataset, DataLoader
from process_data import ProcessData

class Datasets(Dataset):
    """
    Class is utilized to generate specific dataset according to the dataset type
    """
    def __init__(self, config_parameters: dict, ds_type: str):
        """
        Method is utilized as an initializer of the class
        :param config_parameters: all required parameters for the project
        :param ds_type: dataset type that will be processed and collected by the class
        """
        self.data, self.label, self.set_labels = self.get_dataset(config_parameters, ds_type)


    def get_dataset(self, parameters: dict, ds_type: str) -> tuple:
        """
        Method is utilized to collect required dataset in Tensor form
        :param parameters: all required parameters for the project
        :param ds_type: dataset type that will be processed and collected by the class
        :return: tuple which contains data, label and set of labels of the provided dataset
        """
        data_process = ProcessData(parameters, ds_type)
        data = data_process['data']
        shape = data.shape
        print(shape)
        data = data.reshape((shape[0], shape[3], shape[1], shape[2]))
        return torch.FloatTensor(data), \
            torch.LongTensor(data_process['label']), \
            set(data_process['label'])

    def __getitem__(self, item: int) -> dict:
        """
        Method is utilized to get specific data and label information according to the provided item (index in set)
        :param item: integer data that specifies index of the requested data in the set
        :return: dictionary data for requested information
        """
        return {
            'data': self.data[item],
            'label': self.label[item]
        }

    def __len__(self) -> int:
        """
        Method is utilized to get length of the requested dataset
        :return: number of data in the requested set
        """
        return len(self.data)

