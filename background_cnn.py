import numpy as np
import torch
import wandb
import argparse
import utils
from model_wrapper import StandardWrapperBackground
from dataloader import BackgroundDataset, BackgroundDatasetVal
import os
import sys
import random
import pickle


def main():
    commandstring = ''
    for arg in sys.argv:
        if ' ' in arg:
            commandstring += '"{}"  '.format(arg)
        else:
            commandstring += "{}  ".format(arg)

    parser = argparse.ArgumentParser(description='Foveated Convolutional layers')
    parser.add_argument('-dataset', '--dataset', nargs='?', metavar='dataset',
                        default="background", type=str,
                        help='Dataset background")')
    parser.add_argument('-lr', '--lr', nargs='?', metavar='dt', default=0.001, type=float,
                        help='model learning rate; default=0.01')
    parser.add_argument('-opt', '--optimizer', default="adam", type=str,
                        help='Optimizer')
    parser.add_argument('-wrapped_arch', '--wrapped_arch', default="vanilla", type=str,
                        help='Architectures (vanilla, one_layer_global)')
    parser.add_argument('-aggregation_type', '--aggregation_type', default="mean", type=str,
                        help='Region aggreagation {mean, max}')
    parser.add_argument('--grayscale', action='store_true', default=False,
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
    parser.add_argument('--fixed_seed', type=str, default="False",
                        help='For logging the model in tensorboard')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='specify seed (default: 1)')
    parser.add_argument('--batch_dim', type=int, default=32, metavar='batch',
                        help='Batch dims')
    parser.add_argument('--num_workers', type=int, default=0, metavar='S',
                        help='specify number of workers')
    parser.add_argument('--output_channels', type=int, default=1, metavar='out_channels',
                        help='Batch dims')
    parser.add_argument('--kernel', type=int, default=29, metavar='kernel_size',
                        help='Kernel size')
    parser.add_argument('--num_classes', type=int, default=3, metavar='number of classes',
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

    cfg = StandardWrapperBackground.Config()
    cfg.wandb = args.wandb
    cfg.setter(
        vars(args))  # transform args in dict and pass it to the setter - crate a Config instance containing params
    cfg.device = device

    cfg.motion_needed = False
    cfg.foa_flag = False

    cfg.string_command_line = "python " + commandstring.split("/")[-1]

    # creating the model instance
    model = StandardWrapperBackground(cfg)  # instantiate the model wrapper

    ################# DEACTIVATE FOA for standard models ##############
    deactivate_foa = True

    #  Create dataloader class

    dset = BackgroundDatasetVal(version=os.path.join("original_preprocessed", "train"),
                                base_path=os.path.join("data", "task3"),
                                bb_path=os.path.join("original_foa_center", "train"), resize_crop=False,
                                resize_crop_foa=True, num_classes=args.num_classes, deactivate_foa=deactivate_foa)

    folder = os.path.join("data", "task3", "original_preprocessed")
    with open(os.path.join(folder, f'train_indices_{args.num_classes}_classes.pkl'), 'rb') as f:
        train_idx = pickle.load(f)
    with open(os.path.join(folder, f'val_indices_{args.num_classes}_classes.pkl'), 'rb') as f:
        val_idx = pickle.load(f)

    dset_tr = torch.utils.data.Subset(dset, train_idx)
    dset_val = torch.utils.data.Subset(dset, val_idx)

    # val dataset  - notice the activated resize_crop on the foa

    dset_test_original = BackgroundDatasetVal(version=os.path.join("bg_challenge", "original", "val"),
                                              base_path=os.path.join("data", "task3"),
                                              bb_path=os.path.join("val_foa_center"), resize_crop=False,
                                              resize_crop_foa=True, num_classes=args.num_classes,
                                              deactivate_foa=deactivate_foa)

    dset_test_mixedrand = BackgroundDatasetVal(version=os.path.join("bg_challenge", "mixed_rand", "val"),
                                               base_path=os.path.join("data", "task3"),
                                               bb_path=os.path.join("val_foa_center"), resize_crop=False,
                                               resize_crop_foa=True, num_classes=args.num_classes,
                                               deactivate_foa=deactivate_foa)
    dset_test_mixednext = BackgroundDatasetVal(version=os.path.join("bg_challenge", "mixed_next", "val"),
                                               base_path=os.path.join("data", "task3"),
                                               bb_path=os.path.join("val_foa_center"), resize_crop=False,
                                               resize_crop_foa=True, num_classes=args.num_classes,
                                               deactivate_foa=deactivate_foa)
    dset_test_mixedsame = BackgroundDatasetVal(version=os.path.join("bg_challenge", "mixed_same", "val"),
                                               base_path=os.path.join("data", "task3"),
                                               bb_path=os.path.join("val_foa_center"), resize_crop=False,
                                               resize_crop_foa=True, num_classes=args.num_classes,
                                               deactivate_foa=deactivate_foa)

    cfg.foa_options = None  # offline mode

    dset = {"trainset": dset_tr, "valset": dset_val, "test_original": dset_test_original,
            "test_mixedrand": dset_test_mixedrand,
            "test_mixednext": dset_test_mixednext, "test_mixedsame": dset_test_mixedsame
            }

    cfg.foa_options = None  # offline mode


    # # caller model class
    model(dset)  # decidere qui se il foa viene fatto online o meno

    model.train_multiple_test_loop(args.total_epochs)


if __name__ == '__main__':
    main()
