from process_data import ProcessData
from utilities import *
from collect import CollectDataset
from runner import RunModel
from matplotlib import pyplot as plt
from statistics import Statistics
def __main__():
    parameters = collect_parameters()
    # pd = ProcessData(parameters, 'train')
    # for each in pd['data']:
    #     plt.imshow(each)
    #     plt.show()
    #     input()
    stats = Statistics(parameters)
    stats.provide_stats()
    if parameters['train_model']:
        proc_runner = RunModel(parameters)
        proc_runner.train_epoch()
    stats.provide_stats(before=False)
if __name__ == '__main__':
    __main__()