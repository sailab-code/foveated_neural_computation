import numpy as np
import torch
import argparse
import utils
from model_wrapper import StandardMNISTWrapper
from dataloader import MNIST_SETS
import os
import sys
import random


def main():
    commandstring = ''
    for arg in sys.argv:
        if ' ' in arg:
            commandstring += '"{}"  '.format(arg)
        else:
            commandstring += "{}  ".format(arg)

    parser = argparse.ArgumentParser(description='Foveated Convolutional layers')
    parser.add_argument('-dataset', '--dataset', nargs='?', metavar='dataset',
                        default="FashionMNIST_28__MNIST_28-biased-size_200_examples1000", type=str,
                        help='Dataset from the intention-MNIST or stock_MNIST family')
    parser.add_argument('-lr', '--lr', nargs='?', metavar='dt', default=0.001, type=float,
                        help='model learning rate; default=0.01')
    parser.add_argument('-opt', '--optimizer', default="adam", type=str,
                        help='Optimizer')
    parser.add_argument('-wrapped_arch', '--wrapped_arch', default="vanilla", type=str,
                        help='Architectures (vanilla, one_layer_global)')
    parser.add_argument('-aggregation_type', '--aggregation_type', default="max", type=str,
                        help='Region aggreagation {mean, max}')
    parser.add_argument('--grayscale', action='store_true', default=True,
                        help='Force grayscale frames')
    parser.add_argument('-id', '--id', nargs='?', metavar='id', default='', type=str,
                        help='additional id; default=empty string')
    parser.add_argument('-save', '--save_model_flag', action='store_true', default=False,
                        help='save model?; default=1')
    parser.add_argument('-logdir', '--logdir', nargs='?', metavar='logdir', default='tensorboard', type=str,
                        help='directory where the model will be saved; default=tensorboard')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--cuda_dev', type=int, default=0,
                        help='select specific CUDA device for training')
    parser.add_argument('--n_gpu_use', type=int, default=1,
                        help='select number of CUDA device for training')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='logging training status cadency')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='For logging the model in tensorboard')
    parser.add_argument('--fixed_seed',  type=str, default="False",
                        help='For logging the model in tensorboard')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='specify seed (default: 1)')
    parser.add_argument('--batch_dim', type=int, default=32, metavar='batch',
                        help='Batch dims')
    parser.add_argument('--num_workers', type=int, default=0, metavar='S',
                        help='specify number of workers')
    parser.add_argument('--output_channels', type=int, default=128, metavar='out_channels',
                        help='Batch dims')
    parser.add_argument('--kernel', type=int, default=29, metavar='kernel_size',
                        help='Kernel size')
    parser.add_argument('--num_classes', type=int, default=20, metavar='number of classes',
                        help='Number of classes')
    parser.add_argument('--total_epochs', type=int, default=100, metavar='number of epochs',
                        help='Number of epochs')
    parser.add_argument('--wandb', type=str, default="False",
                        help='Log the model in wandb?')
    parser.add_argument('--FLOPS_count', action='store_true', default=False,
                        help='Execution to count FLOPS')

    args = parser.parse_args()
    args.wandb = args.wandb in {'True', 'true'}
    args.fixed_seed = args.fixed_seed in {'True', 'true'}

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not use_cuda:
        args.n_gpu_use = 0

    device = utils.prepare_device(n_gpu_use=args.n_gpu_use, gpu_id=args.cuda_dev)

    # fix  seeds for reproducibility
    if args.fixed_seed:
        SEED = args.seed
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    cfg = StandardMNISTWrapper.Config()
    cfg.wandb = args.wandb
    cfg.setter(
        vars(args))  # transform args in dict and pass it to the setter - crate a Config instance containing params
    cfg.device = device

    cfg.motion_needed = False
    cfg.foa_flag = False

    cfg.string_command_line = "python " + commandstring.split("/")[-1]

    # creating the model instance
    model = StandardMNISTWrapper(cfg)  # instantiate the model wrapper

    #  Create dataloader class

    dset_tr = MNIST_SETS(dataset=os.path.join('data/', args.dataset, "training_samples"),
                         targets=os.path.join('data/', args.dataset, "training_targets"),
                         topK=None)

    dset_val = MNIST_SETS(dataset=os.path.join('data/', args.dataset, "val_samples"),
                          targets=os.path.join('data/', args.dataset, "val_targets"),
                          topK=None)
    dset_test = MNIST_SETS(dataset=os.path.join('data/', args.dataset, "test_samples"),
                           targets=os.path.join('data/', args.dataset, "test_targets"),
                           topK=None)

    cfg.foa_options = None  # offline mode


    dset = {"trainset": dset_tr, "valset": dset_val, "testset": dset_test}
    # # caller model class
    model(dset)  # decidere qui se il foa viene fatto online o meno

    model.train_valid_test_loop(args.total_epochs)


if __name__ == '__main__':
    main()
