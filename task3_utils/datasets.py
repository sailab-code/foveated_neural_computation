# from https://github.com/MadryLab/backgrounds_challenge/blob/master/tools/datasets.py

import torch 
import os
from torchvision import transforms
from task3_utils import folder
from torch.utils.data import DataLoader


def make_loaders_(workers, batch_size, transforms, data_path, dataset, shuffle=True):
    '''
    '''
    print(f"==> Preparing dataset {dataset}..")

    dataset = folder.ImageFolder(root=data_path, transform=transforms)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=shuffle, num_workers=workers, pin_memory=True)

    return loader


class DataSet(object):
    '''
    '''

    def __init__(self, ds_name, data_path, **kwargs):
        """
        """
        required_args = ['num_classes', 'mean', 'std', 'transform_test']
        assert set(kwargs.keys()) == set(required_args), "Missing required args, only saw %s" % kwargs.keys()
        self.ds_name = ds_name
        self.data_path = data_path
        self.__dict__.update(kwargs)

    def make_loaders(self, workers, batch_size, shuffle=False):
        '''
        '''
        transforms = self.transform_test
        return make_loaders_(workers=workers,
                            batch_size=batch_size,
                            transforms=transforms,
                            data_path=self.data_path,
                            dataset=self.ds_name,
                            shuffle=shuffle)

    def get_model(self, arch, pretrained):
        '''
        Args:
            arch (str) : name of architecture
            pretrained (bool): whether to try to load torchvision
                pretrained checkpoint
        Returns:
            A model with the given architecture that works for each
            dataset (e.g. with the right input/output dimensions).
        '''

        raise NotImplementedError


class ImageNet9(DataSet):
    '''
    '''

    def __init__(self, data_path, **kwargs):
        """
        """
        ds_name = 'ImageNet9'
        ds_kwargs = {
            'num_classes': 9,
            'mean': torch.tensor([0.4717, 0.4499, 0.3837]),
            'std': torch.tensor([0.2600, 0.2516, 0.2575]),
            'transform_test': transforms.ToTensor()
        }
        super(ImageNet9, self).__init__(ds_name,
                                        data_path, **ds_kwargs)


class ImageNet(DataSet):
    '''
    '''

    def __init__(self, data_path, **kwargs):
        """
        """
        ds_name = 'ImageNet'
        ds_kwargs = {
            'num_classes': 1000,
            'mean': torch.tensor([0.485, 0.456, 0.406]),
            'std': torch.tensor([0.229, 0.224, 0.225]),
            'transform_test': transforms.ToTensor()
        }
        super(ImageNet, self).__init__(ds_name,
                                       data_path, **ds_kwargs)
