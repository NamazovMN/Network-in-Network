import argparse
import os.path


import torch

def collect_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', default='datasets', type=str, required=False,
                        help='Specifies dataset directory after loading and processing')
    parser.add_argument('--dataset_name', default='mnist', type=str, required=False,
                        choices=['mnist', 'cifar10', 'cifar100', 'svhn'], help='Specifies name of the dataset')
    parser.add_argument('--lambda', default=10, type=float, required=False)
    parser.add_argument('--epsilon', default=1e-4, type=float, required=False)
    parser.add_argument('--s', default=1.0, type=float, required=False)
    parser.add_argument('--global_contrast', default=False, action='store_true', required=False)
    parser.add_argument('--zca_white', default=False, action='store_true', required=False)

    parser.add_argument('--mnist_exp_num', default=13, type=int, required=False)
    parser.add_argument('--cifar10_exp_num', default=13, type=int, required=False)
    parser.add_argument('--cifar100_exp_num', default=13, type=int, required=False)
    parser.add_argument('--svhn_exp_num', default=13, type=int, required=False)
    parser.add_argument('--epochs', default=5, type=int, required=False)
    parser.add_argument('--split', default=0.8, type=float, required=False)
    parser.add_argument('--batch_size', default=32, type=int, required=False)

    parser.add_argument('--mnist_block_out', default=[[192, 160, 96], [192, 192, 192], [192, 192]], required=False, nargs='+')
    parser.add_argument('--mnist_block_kernels', default=[[5, 1, 1], [5, 1, 1], [3, 1, 1]], required=False, nargs='+')
    parser.add_argument('--mnist_block_strides', default=[[1, 1, 1], [1, 1, 1], [1, 1, 1]], required=False, nargs='+')
    parser.add_argument('--mnist_block_paddings', default=[[2, 0, 0], [2, 0, 0], [1, 0, 0]], required=False, nargs='+')
    parser.add_argument('--mnist_dropouts', default=[0.7, 0.7, 0], required=False, nargs='+')
    parser.add_argument('--mnist_pool_choice', default=[True, True, False], required=False, nargs='+')
    parser.add_argument('--mnist_pool_kernels', default=[3, 3, 7], required=False, nargs='+')
    parser.add_argument('--mnist_pool_strides', default=[2, 2, 1], required=False, nargs='+')
    parser.add_argument('--mnist_pool_paddings', default=[1, 1, 0], required=False, nargs='+')
    parser.add_argument('--mnist_batch_norms', default=[False, False, True], required=False, nargs='+')
    parser.add_argument('--mnist_learning_rate', default=0.0001, required=False, nargs='+')

    parser.add_argument('--cifar10_block_out', default=[[192, 160, 96], [192, 192, 192], [192, 192]], required=False, nargs='+')
    parser.add_argument('--cifar10_block_kernels', default=[[5, 1, 1], [5, 1, 1], [3, 1, 1]], required=False, nargs='+')
    parser.add_argument('--cifar10_block_strides', default=[[1, 1, 1], [1, 1, 1], [1, 1, 1]], required=False, nargs='+')
    parser.add_argument('--cifar10_block_paddings', default=[[2, 0, 0], [2, 0, 0], [1, 0, 0]], required=False, nargs='+')
    parser.add_argument('--cifar10_dropouts', default=[0.7, 0.7, 0], required=False, nargs='+')
    parser.add_argument('--cifar10_pool_choice', default=[True, True, False], required=False, nargs='+')
    parser.add_argument('--cifar10_pool_kernels', default=[3, 3, 7], required=False, nargs='+')
    parser.add_argument('--cifar10_pool_strides', default=[2, 2, 1], required=False, nargs='+')
    parser.add_argument('--cifar10_pool_paddings', default=[1, 1, 0], required=False, nargs='+')
    parser.add_argument('--cifar10_batch_norms', default=[False, False, True], required=False, nargs='+')
    parser.add_argument('--cifar10_learning_rate', default=0.0001, required=False, nargs='+')

    parser.add_argument('--cifar100_block_out', default=[[192, 160, 96], [192, 192, 192], [192, 192]], required=False,
                        nargs='+')
    parser.add_argument('--cifar100_block_kernels', default=[[5, 1, 1], [5, 1, 1], [3, 1, 1]], required=False, nargs='+')
    parser.add_argument('--cifar100_block_strides', default=[[1, 1, 1], [1, 1, 1], [1, 1, 1]], required=False, nargs='+')
    parser.add_argument('--cifar100_block_paddings', default=[[2, 0, 0], [2, 0, 0], [1, 0, 0]], required=False,
                        nargs='+')
    parser.add_argument('--cifar100_dropouts', default=[0.7, 0.7, 0], required=False, nargs='+')
    parser.add_argument('--cifar100_pool_choice', default=[True, True, False], required=False, nargs='+')
    parser.add_argument('--cifar100_pool_kernels', default=[3, 3, 7], required=False, nargs='+')
    parser.add_argument('--cifar100_pool_strides', default=[2, 2, 1], required=False, nargs='+')
    parser.add_argument('--cifar100_pool_paddings', default=[1, 1, 0], required=False, nargs='+')
    parser.add_argument('--cifar100_batch_norms', default=[False, False, True], required=False, nargs='+')
    parser.add_argument('--cifar100_learning_rate', default=0.0001, required=False, nargs='+')

    parser.add_argument('--svhn_block_out', default=[[192, 160, 96], [192, 192, 192], [192, 192]], required=False,
                        nargs='+')
    parser.add_argument('--svhn_block_kernels', default=[[5, 1, 1], [5, 1, 1], [3, 1, 1]], required=False, nargs='+')
    parser.add_argument('--svhn_block_strides', default=[[1, 1, 1], [1, 1, 1], [1, 1, 1]], required=False, nargs='+')
    parser.add_argument('--svhn_block_paddings', default=[[2, 0, 0], [2, 0, 0], [1, 0, 0]], required=False,
                        nargs='+')
    parser.add_argument('--svhn_dropouts', default=[0.7, 0.7, 0], required=False, nargs='+')
    parser.add_argument('--svhn_pool_choice', default=[True, True, False], required=False, nargs='+')
    parser.add_argument('--svhn_pool_kernels', default=[3, 3, 8], required=False, nargs='+')
    parser.add_argument('--svhn_pool_strides', default=[2, 2, 1], required=False, nargs='+')
    parser.add_argument('--svhn_pool_paddings', default=[1, 1, 0], required=False, nargs='+')
    parser.add_argument('--svhn_batch_norms', default=[False, False, True], required=False, nargs='+')
    parser.add_argument('--svhn_learning_rate', default=0.0001, required=False, nargs='+')

    parser.add_argument('--train_model', default=False, action='store_true', required=False)
    parser.add_argument('--optimizer', default='Adam', required=False, choices=['Adam', 'SGD'], type=str)
    parser.add_argument('--resume_training', default=False, required=False, action='store_true')

    return parser.parse_args()

def collect_parameters():
    parameters = dict()
    arguments = collect_arguments()
    for argument in vars(arguments):
        parameters[argument] = getattr(arguments, argument)
    parameters['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    return parameters

def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
