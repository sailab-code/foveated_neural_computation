import torch
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split, Subset
from PIL import Image
import os
import torchvision.transforms as transforms
import glob
import numpy as np
import cv2
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
from utils import load_to_npy_gz
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path
from task3_utils.datasets import ImageNet, ImageNet9
from task3_utils.folder import ImageFolder, default_loader, IMG_EXTENSIONS
import xmltodict


class MNIST_FOA(Dataset):

    def __init__(self, dataset, targets, foa_file, fixations, topK=None, every_step=False):
        self.images = load_to_npy_gz(dataset)
        self.targets = load_to_npy_gz(targets)
        self.fixations = fixations
        self.every_step = every_step

        if topK:
            self.images = self.images[:topK]

        self.foa = load_to_npy_gz(foa_file)

        self.length = self.images.shape[0]

        # load one image to know  shape

        # get frame w and h only on the first frame
        self.final_c = self.images.shape[1]
        self.final_h = self.images.shape[2]
        self.final_w = self.images.shape[3]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # in this case, the images have been already preprocessed during the dataset composition (adn the FOA
        # coordinates depend on them, hence we can't change the resolution)

        frame = self.images[idx]
        target = self.targets[idx]

        # compose a batch of frames

        frame_t = torch.from_numpy(frame)
        foa = self.foa[idx][:, :2]
        # foa = foa[:, [1, 0]]  # they were saved in inverse order
        foa_coordinates = torch.from_numpy(foa)

        if self.every_step:
            target = torch.as_tensor(target).repeat(foa_coordinates.shape[0])

        return frame_t, foa_coordinates, target


class TEST_FOA(MNIST_FOA):

    def __init__(self, dataset, targets, foa_file, fixations, topK=None):
        self.images = load_to_npy_gz(dataset)
        self.targets = load_to_npy_gz(targets)
        self.fixations = fixations

        if topK:
            self.images = self.images[:topK]

        self.foa = load_to_npy_gz(foa_file)

        self.length = self.images.shape[0]

        # load one image to know  shape

        # get frame w and h only on the first frame
        self.final_c = self.images.shape[1]
        self.final_h = self.images.shape[2]
        self.final_w = self.images.shape[3]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # in this case, the images have been already preprocessed during the dataset composition (adn the FOA
        # coordinates depend on them, hence we can't change the resolution)

        frame = self.images[idx]
        target = self.targets[idx]

        # compose a batch of frames

        frame_t = torch.from_numpy(frame)
        foa = self.foa[idx][:, :2]
        # foa = foa[:, [1, 0]]  # they were saved in inverse order
        foa_coordinates = torch.from_numpy(foa)

        return frame_t, foa_coordinates, target


class MNIST_SETS(Dataset):

    def __init__(self, dataset, targets, topK=None):
        self.images = load_to_npy_gz(dataset)
        self.targets = load_to_npy_gz(targets)

        if topK:
            self.images = self.images[:topK]

        self.length = self.images.shape[0]

        # load one image to know  shape

        # get frame w and h only on the first frame
        self.final_c = self.images.shape[1]
        self.final_h = self.images.shape[2]
        self.final_w = self.images.shape[3]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # in this case, the images have been already preprocessed during the dataset composition (adn the FOA
        # coordinates depend on them, hence we can't change the resolution)

        frame = self.images[idx]
        target = self.targets[idx]

        # compose a batch of frames

        frame_t = torch.from_numpy(frame)

        return frame_t, target


class MNIST_SETS_FakeFOA(Dataset):

    def __init__(self, dataset, targets, topK=None):
        self.images = load_to_npy_gz(dataset)
        self.targets = load_to_npy_gz(targets)

        if topK:
            self.images = self.images[:topK]

        self.length = self.images.shape[0]

        # load one image to know  shape

        # get frame w and h only on the first frame
        self.final_c = self.images.shape[1]
        self.final_h = self.images.shape[2]
        self.final_w = self.images.shape[3]

        # fake FOA centered in the frame
        self.fake_foa = torch.from_numpy(np.asarray([self.final_h // 2, self.final_w // 2])).unsqueeze(dim=0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # in this case, the images have been already preprocessed during the dataset composition (adn the FOA
        # coordinates depend on them, hence we can't change the resolution)

        frame = self.images[idx]
        target = self.targets[idx]

        # compose a batch of frames
        foa_coordinates = self.fake_foa

        frame_t = torch.from_numpy(frame)

        return frame_t, foa_coordinates, target


## https://pythonawesome.com/pytorch-imagenet1k-loader-with-bounding-boxes/


class CustomDatasetFolder(ImageFolder):
    def __init__(self, root, first_transform=None, second_transform=None, target_transform=None, label_mapping=None,
                 num_classes=None):
        super(ImageFolder, self).__init__(root, loader=default_loader, extensions=IMG_EXTENSIONS,
                                          transform=None,
                                          target_transform=target_transform,
                                          label_mapping=label_mapping,
                                          num_classes=num_classes)
        self.imgs = self.samples
        self.first_transform = first_transform
        self.second_transform = second_transform

    def __getitem__(self, index):
        """
        Modified to return also the path
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        original_dims = sample.size
        transformed_dims = None

        if self.first_transform is not None:
            sample = self.first_transform(sample)
            transformed_dims = sample.size
        if self.second_transform is not None:
            sample = self.second_transform(sample)

        return sample, target, path, original_dims, transformed_dims


class ImageNet9FOA:

    def __init__(self, data_path, resize_crop=True, preprocess=False):
        self.num_classes = 9

        if resize_crop:

            self.first_transform = transforms.Resize(256)

            self.second_transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
            if not preprocess:
                self.second_transform.transforms.append(
                    transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575]))

        else:

            self.first_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            if not preprocess:
                self.first_transform.transforms.append(
                    transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575]))
            self.second_transform = None

        self.data_path = data_path

    def __call__(self, num_classes, *args, **kwargs):
        return CustomDatasetFolder(root=self.data_path, first_transform=self.first_transform,
                                   second_transform=self.second_transform, num_classes=num_classes)


def foa_coordinates_from_BB(expected_class: str, path: str):
    with open(path, 'r') as f:
        data = f.read()
    xml = xmltodict.parse(data)
    assert expected_class == xml["annotation"]["object"]["name"], "Wrong class! Something wrong with the dataset..."
    bb = xml["annotation"]["object"]["bndbox"]
    center_x = (int(bb["xmax"]) + int(bb["xmin"])) // 2
    center_y = (int(bb["ymax"]) + int(bb["ymin"])) // 2

    w = xml["annotation"]["size"]["width"]
    h = xml["annotation"]["size"]["height"]

    return center_y, center_x, w, h


##################################################################################################################

class ImagenetBB:
    def __init__(self, bb_path, resize_crop=True):
        self.bb_path = bb_path
        self.resize_crop = resize_crop

    def rescale_bb(self, foa, original_dims, new_dims):
        """
        Method to rescale the foa coordinates from the original Imagenet size, to the new coordinates corresponding
        to a Resize(256) followed by a CenterCrop(224)
        Note that the Resize(256) produces images having the smallest dim equal to 256, and the other can vary.
        """
        # compute relative position of foa inside the frame
        foa_y_relative = foa[0] / original_dims[1]
        foa_x_relative = foa[1] / original_dims[0]

        # we start from the  frame resized with Resize(256)
        w, h = new_dims
        # here the dimensions after the CenterCrop(224) -- can be modified
        w_prime, h_prime = 224, 224

        # compute new foa relative positions - here the formula

        new_foa_x_relative = (w * foa_x_relative) / w_prime - (w - w_prime) / (2 * w_prime)
        new_foa_y_relative = (h * foa_y_relative) / h_prime - (h - h_prime) / (2 * h_prime)

        # compute new foa positions, multipliying the relative positions to the new coordinates

        new_foa_y = round(new_foa_y_relative * h_prime)
        new_foa_x = round(new_foa_x_relative * w_prime)

        return [new_foa_y, new_foa_x]

    def __call__(self, path, original_dims=None, new_dims=None):
        filename = Path(path).stem
        imagenet_class = filename.split("_")[0]

        # returns a list [[xmin, xmax, ymin, ymax]]
        foay, foax, _, _ = foa_coordinates_from_BB(expected_class=imagenet_class,
                                                   path=os.path.join(self.bb_path, imagenet_class, filename) + ".xml")
        foa = [foay, foax]
        if self.resize_crop:
            foa = self.rescale_bb(foa, original_dims, new_dims)

        return torch.from_numpy(np.asarray(foa))


class ImageNetBBVal(ImagenetBB):
    def __init__(self, bb_path, resize_crop=True, version=None, foa_centered=False):
        super().__init__(bb_path, resize_crop=resize_crop)
        self.version = version
        self.foa_centered = foa_centered

    def __call__(self, path, original_dims=None, new_dims=None):
        if self.foa_centered:
            # return the center of the image
            return torch.from_numpy(np.asarray([112, 112]))
        else:
            filename = Path(path).stem
            bg_challenge_class_name = Path(path).parents[0].name
            if "original" in self.version:
                # different name format in case of  original
                pass
            else:
                foreground_img = filename.split("_bg")[0]
                class_name = foreground_img.split("_")[1]
                filename = foreground_img[3:]  # exclude the string "fg_"

            # returns a list [[xmin, xmax, ymin, ymax]]
            foa = np.load(os.path.join(self.bb_path, bg_challenge_class_name, filename) + ".npy")

            foa = np.clip(foa, 0, 223)

            return torch.from_numpy(foa)


class BackgroundDataset:

    def __init__(self, version, base_path, bb_path=None, resize_crop=True, resize_crop_foa=False, preprocess=False,
                 num_classes=9):
        self.img_object = ImageNet9FOA(os.path.join(base_path, version), resize_crop=resize_crop, preprocess=preprocess)
        self.img_dataset = self.img_object(num_classes=num_classes)
        self.base_path = base_path
        self.bb_object = ImagenetBB(os.path.join(base_path, bb_path), resize_crop=resize_crop_foa)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, idx):
        # in this case, the images have been already preprocessed during the dataset composition (adn the FOA
        # coordinates depend on them, hence we can't change the resolution)

        # use the dataset getitem
        frame, target, path, original_dims, new_dims = self.img_dataset[idx]
        foa_coordinates = self.bb_object(path, original_dims, new_dims)
        if self.preprocess:
            return frame, foa_coordinates, target, path
        else:
            return frame, foa_coordinates, target


class BackgroundDatasetVal(BackgroundDataset):
    def __init__(self, version, base_path, bb_path=None, resize_crop=True, resize_crop_foa=False, preprocess=False,
                 num_classes=None, deactivate_foa=False, foa_centered=False):
        # used for standard nets
        self.deactivate_foa = deactivate_foa
        self.preprocess = preprocess
        self.num_classes = num_classes
        self.img_object = ImageNet9FOA(os.path.join(base_path, version), resize_crop=resize_crop, preprocess=preprocess)
        self.img_dataset = self.img_object(num_classes=num_classes)
        self.base_path = base_path
        self.bb_object = ImageNetBBVal(os.path.join(base_path, bb_path), resize_crop=resize_crop_foa, version=version,
                                       foa_centered=foa_centered)

    def __getitem__(self, idx):
        # in this case, the images have been already preprocessed during the dataset composition (adn the FOA
        # coordinates depend on them, hence we can't change the resolution)

        # use the dataset getitem
        frame, target, path, original_dims, new_dims = self.img_dataset[idx]
        if self.deactivate_foa:
            return frame, target
        else:
            foa_coordinates = self.bb_object(path, original_dims, new_dims)

            if self.preprocess:
                return frame, foa_coordinates, target, path
            else:
                return frame, foa_coordinates, target


class FramesandFOA(Dataset):

    def __init__(self, root_dir, foa_file, force_gray=True):
        self.root_dir = root_dir
        self.files = glob.glob(root_dir + os.sep + "**" + os.sep + "*.png", recursive=True)
        self.files = sorted(self.files)
        if not os.path.exists(foa_file):
            raise IOError("Cannot find the specified FOA file: ", foa_file)
        else:
            self.foa = np.loadtxt(foa_file, delimiter=",")

        self.force_gray = force_gray
        self.length = len(self.files)

        # load one image to know  shape

        # get frame w and h only on the first frame
        frame = cv2.imread(self.files[0])
        self.final_c = frame.shape[2]
        self.final_h = frame.shape[0]
        self.final_w = frame.shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # in this case, the images have been already preprocessed during the dataset composition (adn the FOA
        # coordinates depend on them, hence we can't change the resolution)

        frame = cv2.imread(self.files[idx])

        # get frame w and h only on the first frame

        if self.force_gray and frame.shape[2] > 1:
            frame = np.reshape(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (frame.shape[0], frame.shape[1], 1))

        foa = self.foa[idx, :]
        foa_coordinates = torch.tensor([foa[0], foa[1]], dtype=torch.long)

        frame = torch.from_numpy(frame.transpose(2, 0, 1)).float().div_(255.0)
        target = torch.zeros((frame.shape[0]))
        return frame, foa_coordinates, target


class TestSet(Dataset):
    def __init__(self, root_dir, foa_file, force_gray=True):
        self.root_dir = root_dir
        self.files = glob.glob(root_dir + os.sep + "**" + os.sep + "*.png", recursive=True)
        self.files = sorted(self.files)

        self.force_gray = force_gray
        self.length = len(self.files)

        # load one image to know  shape

        # get frame w and h only on the first frame
        frame = cv2.imread(self.files[0])
        self.final_c = frame.shape[2]
        self.final_h = frame.shape[0]
        self.final_w = frame.shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # in this case, the images have been already preprocessed during the dataset composition (adn the FOA
        # coordinates depend on them, hence we can't change the resolution)

        frame = cv2.imread(self.files[idx])

        # get frame w and h only on the first frame

        if self.force_gray and frame.shape[2] > 1:
            frame = np.reshape(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (frame.shape[0], frame.shape[1], 1))

        foa_coordinates = torch.randint(low=0, high=max(self.final_w, self.final_h), size=(10, 2))

        frame = torch.from_numpy(frame.transpose(2, 0, 1)).float().div_(255.0)
        target = torch.zeros((frame.shape[0]))
        return frame, foa_coordinates, target


class RepeatedFrames(Dataset):
    def __init__(self, folder="data/car_frame", repeat_frame=True, repeat_times=1000, grayscale=False, height=-1,
                 width=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.folder = folder
        self.repeat_frame = repeat_frame
        self.repeat_times = repeat_times

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        if grayscale:
            self.transform.transforms.append(transforms.Grayscale())
        if height > 0 and width > 0:  # TODO maybe improve here when only one provided...
            self.transform.transforms.append(transforms.Resize((height, width)))

        self.image_files = os.listdir(self.folder)
        images = []
        for i in self.image_files:
            image = Image.open(os.path.join(self.folder, i))
            if self.transform:
                image = self.transform(image)
            if repeat_frame:
                for i in range(repeat_times):
                    images.append(image)
            else:
                images.append(image)

        x = torch.stack(images)

        self.data = x
        gray = transforms.Grayscale()
        self.data_gray = x if grayscale else gray(x)
        self.targets = torch.zeros_like(x)
        self.original_shape = self.data.shape
        self.final_c = self.original_shape[1]
        self.final_h = self.original_shape[2]
        self.final_w = self.original_shape[3]

    def __getwhole__(self, device, target_transform=None):
        """

        :param device:
        :param target_transform: transform to apply on the whole target tensor (to one hot)
        :return:
        """
        # Put both data and targets on GPU in advance

        return self.data.to(device), self.targets.to(device), self.data_gray.to(device)


class FastMNIST(MNIST):
    def __init__(self, root="MNIST", download=True, white_background=False, normalize=False, *args, **kwargs):
        super().__init__(root, download=True, *args, **kwargs)

        self.original_shape = self.data.shape  # to get original data shape
        self.data = self.data.unsqueeze(1).float()  # adding color channel (to be consistent with other image datasets)

        # Scale data to [0,1]
        if normalize:
            self.data = self.data.div(255.)
            # Normalize it with the usual MNIST mean and std
            # self.data = self.data.sub_(0.1307).div_(0.3081)

        if white_background:
            if normalize:
                self.data = 1. - self.data
            else:
                self.data = 255. - self.data  # TODO white background

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target

    def __getwhole__(self, device, target_transform):
        """

        :param device:
        :param target_transform: transform to apply on the whole target tensor (to one hot)
        :return:
        """
        # Put both data and targets on GPU in advance

        return self.data.to(device), target_transform(self.targets).float().to(device)

    def __getwhole_flattened__(self, device, target_transform=None):
        """

        :param device:
        :param target_transform: transform to apply on the whole target tensor (to one hot)
        :return:
        """
        # Put both data and targets on GPU in advance
        if target_transform is not None:

            return self.data.flatten(start_dim=1).to(device), target_transform(self.targets).float().to(device)
        else:
            return self.data.flatten(start_dim=1).to(device), self.targets.float().to(device)


# code from https://github.com/bentrevett/recurrent-attention-model/blob/master/data_loader.py


def get_data(which, batch_size, root='data'):
    # get train data

    # define transforms
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # get train data w/ transforms
    train_data = getattr(datasets, which)(root=root,
                                          train=True,
                                          download=True,
                                          transform=data_transforms)

    # get test data w/ transforms
    test_data = getattr(datasets, which)(root=root,
                                         train=False,
                                         download=True,
                                         transform=data_transforms)

    # load train and test iterators
    train_iterator = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, )
    test_iterator = data.DataLoader(test_data, batch_size=batch_size)

    return train_iterator, test_iterator


def get_translated_data(which, translated_size, batch_size, root='data', biased=False):
    # hard coded for MNIST style datasets
    image_size = 28

    # define transform that does nothing
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    norm_transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = getattr(datasets, which)(root=root,
                                          train=True,
                                          download=True,
                                          transform=data_transforms)

    # get test data w/ transforms
    test_data = getattr(datasets, which)(root=root,
                                         train=False,
                                         download=True,
                                         transform=data_transforms)

    # load collator
    collator = TranslatedCollator(image_size, translated_size, norm_transform, biased="up")
    collator_tst = TranslatedCollator(image_size, translated_size, norm_transform, biased="down")

    # load train and test iterators
    train_iterator = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collator.collate)
    test_iterator = data.DataLoader(test_data, batch_size=batch_size, collate_fn=collator_tst.collate)

    return train_iterator, test_iterator


def get_cluttered_data(which, translated_size, n_clutter, clutter_size, batch_size, root='data', biased=False):
    # hard coded for MNIST style datasets
    image_size = 28

    # define transform that does nothing
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    norm_transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # get train data w/ transforms
    train_data = getattr(datasets, which)(root=root,
                                          train=True,
                                          download=True,
                                          transform=data_transforms)

    # get test data w/ transforms
    test_data = getattr(datasets, which)(root=root,
                                         train=False,
                                         download=True,
                                         transform=data_transforms)

    # load collator

    collator = ClutteredCollator(image_size, translated_size, n_clutter, clutter_size, norm_transform, biased="up")
    collator_tst = ClutteredCollator(image_size, translated_size, n_clutter, clutter_size, norm_transform,
                                     biased="down")

    # load train and test iterators
    train_iterator = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collator.collate)
    test_iterator = data.DataLoader(test_data, batch_size=batch_size, collate_fn=collator_tst.collate)

    return train_iterator, test_iterator


class TranslatedCollator:
    def __init__(self, image_size, translated_size, norm_transform, biased):
        self.image_size = image_size
        self.translated_size = translated_size
        self.data_transforms = norm_transform
        self.biased = biased

    def collate(self, batch):
        images, labels = zip(*batch)
        batch_size = len(images)
        background = torch.zeros(batch_size, 1, self.translated_size, self.translated_size)
        image_pos = torch.randint(0, self.translated_size - self.image_size, (batch_size, 2))
        if self.biased == "up":
            image_pos[:, 1] = torch.randint(0, (self.translated_size - self.image_size) // 4, (batch_size,))
        elif self.biased == "down":
            image_pos[:, 1] = torch.randint(int((self.translated_size - self.image_size) / (4 / 3)),
                                            self.translated_size - self.image_size, (batch_size,))
        for i, image in enumerate(images):
            background[i, :, image_pos[i][1]:image_pos[i][1] + self.image_size,
            image_pos[i][0]:image_pos[i][0] + self.image_size] = image
        labels = torch.LongTensor(labels)

        return self.data_transforms(background), labels


class TranslatedCollatorOrigin:
    def __init__(self, image_size, translated_size):
        self.image_size = image_size
        self.translated_size = translated_size

    def collate(self, batch):
        images, labels = zip(*batch)
        batch_size = len(images)
        background = torch.zeros(batch_size, 1, self.translated_size, self.translated_size)
        image_pos = torch.randint(0, self.translated_size - self.image_size, (batch_size, 2))
        for i, image in enumerate(images):
            background[i, :, image_pos[i][1]:image_pos[i][1] + self.image_size,
            image_pos[i][0]:image_pos[i][0] + self.image_size] = image
        labels = torch.LongTensor(labels)

        return background, labels


class ClutteredCollator:
    def __init__(self, image_size, translated_size, n_clutter, clutter_size, norm_transform, biased):
        self.image_size = image_size
        self.translated_size = translated_size
        self.n_clutter = n_clutter
        self.clutter_size = clutter_size
        self.data_transforms = norm_transform
        self.biased = biased

    def collate(self, batch):
        images, labels = zip(*batch)
        batch_size = len(images)
        background = torch.zeros(batch_size, 1, self.translated_size, self.translated_size)
        clutter_slice = torch.randint(0, self.image_size - self.clutter_size, (batch_size, self.n_clutter, 2))
        clutter_pos = torch.randint(0, self.translated_size - self.clutter_size, (batch_size, self.n_clutter, 2))
        image_pos = torch.randint(0, self.translated_size - self.image_size, (batch_size, 2))
        if self.biased == "up":
            image_pos[:, 1] = torch.randint(0, (self.translated_size - self.image_size) // 4, (batch_size,))
        elif self.biased == "down":
            image_pos[:, 1] = torch.randint(int((self.translated_size - self.image_size) / (4 / 3)),
                                            self.translated_size - self.image_size, (batch_size,))
        for i, image in enumerate(images):
            for j in range(self.n_clutter):
                clutter_full = random.choice(images)
                clutter = clutter_full[:, clutter_slice[i][j][1]:clutter_slice[i][j][1] + self.clutter_size,
                          clutter_slice[i][j][0]:clutter_slice[i][j][0] + self.clutter_size]
                background[i, :, clutter_pos[i][j][1]:clutter_pos[i][j][1] + self.clutter_size,
                clutter_pos[i][j][0]:clutter_pos[i][j][0] + self.clutter_size] = clutter
            background[i, :, image_pos[i][1]:image_pos[i][1] + self.image_size,
            image_pos[i][0]:image_pos[i][0] + self.image_size] = image
        labels = torch.LongTensor(labels)
        return self.data_transforms(background), labels


class ClutteredCollatorOrigin:
    def __init__(self, image_size, translated_size, n_clutter, clutter_size):
        self.image_size = image_size
        self.translated_size = translated_size
        self.n_clutter = n_clutter
        self.clutter_size = clutter_size

    def collate(self, batch):
        images, labels = zip(*batch)
        batch_size = len(images)
        background = torch.zeros(batch_size, 1, self.translated_size, self.translated_size)
        clutter_slice = torch.randint(0, self.image_size - self.clutter_size, (batch_size, self.n_clutter, 2))
        clutter_pos = torch.randint(0, self.translated_size - self.clutter_size, (batch_size, self.n_clutter, 2))
        image_pos = torch.randint(0, self.translated_size - self.image_size, (batch_size, 2))
        for i, image in enumerate(images):
            for j in range(self.n_clutter):
                clutter_full = random.choice(images)
                clutter = clutter_full[:, clutter_slice[i][j][1]:clutter_slice[i][j][1] + self.clutter_size,
                          clutter_slice[i][j][0]:clutter_slice[i][j][0] + self.clutter_size]
                background[i, :, clutter_pos[i][j][1]:clutter_pos[i][j][1] + self.clutter_size,
                clutter_pos[i][j][0]:clutter_pos[i][j][0] + self.clutter_size] = clutter
            background[i, :, image_pos[i][1]:image_pos[i][1] + self.image_size,
            image_pos[i][0]:image_pos[i][0] + self.image_size] = image
        labels = torch.LongTensor(labels)
        return background, labels


def get_split_set(config, root="data"):
    which = config.middle_task
    # which, middle_task_size, n_clutter, clutter_size, batch_size, biased = config.middle_task, config.middle_task_size,
    # config.n_clutter,  config.clutter_size, config.batch_size, config.biased, config.
    # hard coded for MNIST style datasets

    # define transform that does nothing

    whole_data = get_and_fuse_sets(which=which, root=root)

    # pick random examples from the whole dataset - the double amount so that we can build the final dataset

    indices = torch.randint(low=0, high=len(whole_data), size=(config.dataset_full_size * 2,))

    train_ratio = config.train_val_test_ratio[0] / 100
    validation_ratio = config.train_val_test_ratio[1] / 100
    test_ratio = config.train_val_test_ratio[2] / 100

    # train is now 75% of the entire data set
    # the _junk suffix means that we drop that variable completely
    idx_train, idx_test = train_test_split(indices, test_size=1 - train_ratio)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    idx_val, idx_test = train_test_split(idx_test, test_size=test_ratio / (test_ratio + validation_ratio))

    loaders_tr = get_dataloader(whole_data, idx_train, config)
    loaders_val = get_dataloader(whole_data, idx_val, config)
    loaders_test = get_dataloader(whole_data, idx_test, config)

    return loaders_tr, loaders_val, loaders_test


def get_and_fuse_sets(which, root):
    # define transform
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    # get train data w/ transforms
    train_data = getattr(datasets, which)(root=root,
                                          train=True,
                                          download=True,
                                          transform=data_transforms)

    # get test data w/ transforms
    test_data = getattr(datasets, which)(root=root,
                                         train=False,
                                         download=True,
                                         transform=data_transforms)

    # merge train and test

    whole_data = ConcatDataset([train_data, test_data])

    return whole_data


def get_dataloader(input_data, indices, config):
    whole_data = Subset(input_data, indices)
    # now split into two sets (middle and border)

    data_size = len(whole_data) // 2

    middle, periphery = random_split(whole_data, [data_size, data_size],  # limit the amount of gathered data
                                     generator=torch.Generator().manual_seed(config.seed))

    middle_loader = data.DataLoader(middle, batch_size=config.batch_size, shuffle=True)
    periphery_loader = data.DataLoader(periphery, batch_size=config.batch_size, shuffle=True)

    return middle_loader, periphery_loader


def get_double_set(config, root="data"):
    which = config.middle_task
    which_per = config.peripheral_task

    whole_middle = get_and_fuse_sets(which=which, root=root)
    whole_per = get_and_fuse_sets(which=which_per, root=root)

    # pick random examples from the whole dataset - the double amount so that we can build the final dataset

    indices_middle = torch.randint(low=0, high=len(whole_middle), size=(config.dataset_full_size,))
    indices_per = torch.randint(low=0, high=len(whole_per), size=(config.dataset_full_size,))

    train_ratio = config.train_val_test_ratio[0] / 100
    validation_ratio = config.train_val_test_ratio[1] / 100
    test_ratio = config.train_val_test_ratio[2] / 100

    # train is now 75% of the entire data set
    # the _junk suffix means that we drop that variable completely
    idx_train_middle, idx_test_middle = train_test_split(indices_middle, test_size=1 - train_ratio)
    idx_train_per, idx_test_per = train_test_split(indices_per, test_size=1 - train_ratio)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    idx_val_middle, idx_test_middle = train_test_split(idx_test_middle,
                                                       test_size=test_ratio / (test_ratio + validation_ratio))
    idx_val_per, idx_test_per = train_test_split(idx_test_per, test_size=test_ratio / (test_ratio + validation_ratio))

    loaders_tr = get_dataloader_mixed(whole_middle, idx_train_middle, whole_per, idx_train_per, config)
    loaders_val = get_dataloader_mixed(whole_middle, idx_val_middle, whole_per, idx_val_per, config)
    loaders_test = get_dataloader_mixed(whole_middle, idx_test_middle, whole_per, idx_test_per, config)

    return loaders_tr, loaders_val, loaders_test


def get_dataloader_mixed(input_data_middle, indices_middle, input_data_per, indices_per, config):
    middle = Subset(input_data_middle, indices_middle)
    periphery = Subset(input_data_per, indices_per)
    # now split into two sets (middle and border)

    middle_loader = data.DataLoader(middle, batch_size=config.batch_size, shuffle=True)
    periphery_loader = data.DataLoader(periphery, batch_size=config.batch_size, shuffle=True)

    return middle_loader, periphery_loader


def draw_circle(size):
    radius = size // 2
    region_x, region_y = torch.meshgrid(
        torch.arange(start=-radius + 1, end=radius + 1, dtype=torch.long),
        torch.arange(start=-radius + 1, end=radius + 1, dtype=torch.long), indexing="ij")

    valid = torch.sqrt(region_x ** 2 + region_y ** 2) < radius
    intent = valid.int()
    return intent


def draw_rect(size, border_size=2):
    radius = size // 2
    region_x, region_y = torch.meshgrid(
        torch.arange(start=-radius + 1, end=radius + 1, dtype=torch.long),
        torch.arange(start=-radius + 1, end=radius + 1, dtype=torch.long), indexing="ij")

    valid_x = torch.logical_and(torch.abs(region_x) < radius + 1, torch.abs(region_x) >= radius - border_size + 1)
    valid_y = torch.logical_and(torch.abs(region_y) < radius + 1, torch.abs(region_y) >= radius - border_size + 1)

    valid = torch.logical_or(valid_x, valid_y)
    intent = valid.int()
    return intent


def create_sets_from_tasks(middle_loader, periphery_loader, config, mode="train", border=5, multi_label=False):
    dataset_stats = {str(i): 0 for i in range(10)}
    if config.intent_size is not None:
        dataset_stats["middle"] = 0
        dataset_stats['periphery'] = 0

    frame_size = config.frame_size
    frame_center = frame_size // 2

    example_list = []
    target_list = []

    norm_transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if config.intent_size is not None:
        circle_intent = draw_circle(config.intent_size)
        square_intent = draw_rect(config.intent_size)

    periphery_iterator = iter(periphery_loader)

    # go over the two splits and compose them into one
    with tqdm(middle_loader, unit="batch") as tepoch_t:
        for i_middle in tepoch_t:
            middle_input, middle_target = i_middle
            # get also the next batch from periphery
            periphery_input, periphery_target = next(periphery_iterator)

            batch_size = len(middle_input)  # effective batch_size

            # build common black background
            background = torch.zeros(batch_size, 1, config.frame_size, config.frame_size)

            # position the middle task at the center of the image
            c_p_top = frame_center - config.middle_task_size // 2
            c_p_bottom = frame_center + config.middle_task_size // 2
            background[:, :, c_p_top:c_p_bottom, c_p_top:c_p_bottom] = middle_input

            # position the peripheral task in the top frame  quarter

            if mode == "train" and config.biased:
                periphery_pos_x = torch.randint(border, (config.frame_size - config.peripheral_task_size - border),
                                                (batch_size, 1))
                # limit y dim in the top part
                periphery_pos_y = torch.randint(border, (config.frame_size - config.peripheral_task_size) // 8,
                                                (batch_size, 1))
                periphery_pos = torch.hstack((periphery_pos_x, periphery_pos_y))

                if config.intent_size is not None:
                    intent_pos_x = torch.randint(border, config.frame_size - config.intent_size - border,
                                                 (batch_size, 1))
                    intent_pos_y = torch.randint(int((config.frame_size - config.intent_size) / (9 / 8)),
                                                 config.frame_size - config.intent_size - border, (batch_size, 1))
                    intent_pos = torch.hstack((intent_pos_x, intent_pos_y))

            # swap in case train/test biased
            if mode == "test" and config.biased:
                periphery_pos_x = torch.randint(border, (config.frame_size - config.peripheral_task_size - border),
                                                (batch_size, 1))
                # limit y dim in the top part
                periphery_pos_y = torch.randint(int((config.frame_size - config.peripheral_task_size) / (9 / 8)),
                                                config.frame_size - config.peripheral_task_size - border,
                                                (batch_size, 1))
                periphery_pos = torch.hstack((periphery_pos_x, periphery_pos_y))
                if config.intent_size is not None:
                    intent_pos_x = torch.randint(border, config.frame_size - config.intent_size - border,
                                                 (batch_size, 1))
                    intent_pos_y = torch.randint(border, (config.frame_size - config.intent_size) // 8,
                                                 (batch_size, 1))
                    intent_pos = torch.hstack((intent_pos_x, intent_pos_y))

            if config.intent_size is not None:
                intent_choice = torch.randint(0, 2, (batch_size,))

            # generate both intents for this batch

            targets_batch_list = []

            for i, image in enumerate(periphery_input):
                background[i, :, periphery_pos[i, 1]:periphery_pos[i, 1] + config.peripheral_task_size,
                periphery_pos[i, 0]:periphery_pos[i, 0] + config.peripheral_task_size] = image

                if config.intent_size is not None:
                    if intent_choice[i] == 0:
                        # selected periphery task
                        background[i, :, intent_pos[i, 1]:intent_pos[i, 1] + config.intent_size,
                        intent_pos[i, 0]:intent_pos[i, 0] + config.intent_size] = circle_intent
                        targets_batch_list.append(middle_target[None, i])
                        if config.stats:
                            dataset_stats[str(middle_target[i].item())] += 1
                            if config.intent_size is not None:
                                dataset_stats["middle"] += 1

                    else:
                        background[i, :, intent_pos[i, 1]:intent_pos[i, 1] + config.intent_size,
                        intent_pos[i, 0]:intent_pos[i, 0] + config.intent_size] = square_intent
                        targets_batch_list.append(periphery_target[None, i])
                        if config.stats:
                            dataset_stats[str(periphery_target[i].item())] += 1
                            if config.intent_size is not None:
                                dataset_stats["periphery"] += 1
            if multi_label:
                availability_periphery_target = torch.where(periphery_target > 4, 1, 0)
                # append multi-label target (if periphery target <5, per. label is 0, else 1)
                targets_batch_list.append(
                    middle_target + 10 * availability_periphery_target)
                # torch.hstack((middle_target.unsqueeze(1), availability_periphery_target.unsqueeze(1))))

            targets_batch_list = torch.cat(targets_batch_list)

            # normalize the whole frame
            example_list.append(norm_transform(background))
            target_list.append(targets_batch_list)

    return torch.cat(example_list).numpy(), torch.cat(target_list).numpy(), dataset_stats


def create_test_dual_frame_dataset(middle_loader, periphery_loader, config, mode="train", border=5, multi_label=False):
    if config.four:
        return create_test_four_frame_dataset(middle_loader, periphery_loader, config, mode="train", border=5,
                                              multi_label=False)

    frame_size = config.frame_size
    frame_center = frame_size // 2

    example_list = []
    target_list = []

    norm_transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if config.intent_size is not None:
        circle_intent = draw_circle(config.intent_size)
        square_intent = draw_rect(config.intent_size)

    periphery_iterator = iter(periphery_loader)
    middle_iterator = iter(middle_loader)
    periphery_input, periphery_target = next(periphery_iterator)
    middle_input, middle_target = next(middle_iterator)

    # select only first samples
    periphery_input, periphery_target = periphery_input[0], periphery_target[0]
    middle_input, middle_target = middle_input[0], middle_target[0]

    batch_size = 2  # 2 examples in the dataset
    # build common black background
    background = torch.zeros(batch_size, 1, config.frame_size, config.frame_size)

    # position the middle task at the center of the image
    c_p_top = frame_center - config.middle_task_size // 2
    c_p_bottom = frame_center + config.middle_task_size // 2

    # first example
    background[0, :, c_p_top:c_p_bottom, c_p_top:c_p_bottom] = middle_input

    # position the peripheral task in the top frame  quarter

    if mode == "train" and config.biased:
        periphery_pos_x = torch.randint(border, (config.frame_size - config.peripheral_task_size - border),
                                        (batch_size, 1))
        # limit y dim in the top part
        periphery_pos_y = torch.randint(border, (config.frame_size - config.peripheral_task_size) // 8,
                                        (batch_size, 1))
        periphery_pos = torch.hstack((periphery_pos_x, periphery_pos_y))

        if config.intent_size is not None:
            intent_pos_x = torch.randint(border, config.frame_size - config.intent_size, (batch_size, 1))
            intent_pos_y = torch.randint(int((config.frame_size - config.intent_size) / (9 / 8)),
                                         config.frame_size - config.intent_size - border, (batch_size, 1))
            intent_pos = torch.hstack((intent_pos_x, intent_pos_y))

    # swap in case train/test biased
    if mode == "test" and config.biased:
        periphery_pos_x = torch.randint(border, (config.frame_size - config.peripheral_task_size - border),
                                        (batch_size, 1))
        # limit y dim in the top part
        periphery_pos_y = torch.randint(int((config.frame_size - config.peripheral_task_size) / (9 / 8)),
                                        config.frame_size - config.peripheral_task_size - border,
                                        (batch_size, 1))
        periphery_pos = torch.hstack((periphery_pos_x, periphery_pos_y))
        if config.intent_size is not None:
            intent_pos_x = torch.randint(config.frame_size - config.intent_size - border, (batch_size, 1))
            intent_pos_y = torch.randint(border, (config.frame_size - config.intent_size) // 8,
                                         (batch_size, 1))
            intent_pos = torch.hstack((intent_pos_x, intent_pos_y))

    if config.intent_size is not None:
        intent_choice = torch.randint(0, 2, (1,))

    # generate both intents for this batch

    targets_batch_list = []

    # still first example
    i = 0
    image = periphery_input
    background[i, :, periphery_pos[i, 1]:periphery_pos[i, 1] + config.peripheral_task_size,
    periphery_pos[i, 0]:periphery_pos[i, 0] + config.peripheral_task_size] = image

    if config.intent_size is not None:
        if intent_choice == 0:
            # selected periphery task
            background[i, :, intent_pos[i, 1]:intent_pos[i, 1] + config.intent_size,
            intent_pos[i, 0]:intent_pos[i, 0] + config.intent_size] = circle_intent
            targets_batch_list.append(middle_target.unsqueeze(dim=0))

        else:
            background[i, :, intent_pos[i, 1]:intent_pos[i, 1] + config.intent_size,
            intent_pos[i, 0]:intent_pos[i, 0] + config.intent_size] = square_intent
            targets_batch_list.append(periphery_target.unsqueeze(dim=0))

    else:
        # target 0 for first example
        targets_batch_list.append(torch.zeros((1), dtype=torch.long))

    # second example

    background[1, :, c_p_top:c_p_bottom, c_p_top:c_p_bottom] = periphery_input
    i = 1
    image = middle_input
    background[i, :, periphery_pos[i, 1]:periphery_pos[i, 1] + config.peripheral_task_size,
    periphery_pos[i, 0]:periphery_pos[i, 0] + config.peripheral_task_size] = image

    # target/intent positioning
    if config.intent_size is not None:
        if intent_choice == 0:
            # selected periphery task
            background[i, :, intent_pos[i, 1]:intent_pos[i, 1] + config.intent_size,
            intent_pos[i, 0]:intent_pos[i, 0] + config.intent_size] = circle_intent
            targets_batch_list.append(periphery_target.unsqueeze(dim=0))

        else:
            background[i, :, intent_pos[i, 1]:intent_pos[i, 1] + config.intent_size,
            intent_pos[i, 0]:intent_pos[i, 0] + config.intent_size] = square_intent
            targets_batch_list.append(middle_target.unsqueeze(dim=0))

    else:
        # target 0 for first example
        targets_batch_list.append(torch.ones((1), dtype=torch.long))

    targets_batch_list = torch.cat(targets_batch_list)

    # normalize the whole frame
    example_list.append(norm_transform(background))

    return torch.cat(example_list).numpy(), targets_batch_list.numpy(), None


def create_test_four_frame_dataset(middle_loader, periphery_loader, config, mode="train", border=5, multi_label=False):
    frame_size = config.frame_size
    frame_center = frame_size // 2

    example_list = []
    target_list = []

    norm_transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if config.intent_size is not None:
        circle_intent = draw_circle(config.intent_size)
        square_intent = draw_rect(config.intent_size)

    periphery_iterator = iter(periphery_loader)
    middle_iterator = iter(middle_loader)
    periphery_input, periphery_target = next(periphery_iterator)
    middle_input, middle_target = next(middle_iterator)

    # select only first samples
    periphery_input, periphery_target = periphery_input[0], periphery_target[0]
    middle_input, middle_target = middle_input[0], middle_target[0]

    batch_size = 4  # 2 examples in the dataset
    # build common black background
    background = torch.zeros(batch_size, 1, config.frame_size, config.frame_size)

    # position the middle task at the center of the image
    c_p_top = frame_center - config.middle_task_size // 2
    c_p_bottom = frame_center + config.middle_task_size // 2

    # first example

    background[0, :, c_p_top:c_p_bottom, c_p_top:c_p_bottom] = middle_input
    background[1, :, c_p_top:c_p_bottom, c_p_top:c_p_bottom] = middle_input

    # position the peripheral task in the top frame  quarter

    if mode == "train" and config.biased:
        periphery_pos_x = torch.randint(border, (config.frame_size - config.peripheral_task_size - border),
                                        (batch_size, 1))
        # limit y dim in the top part
        periphery_pos_y = torch.randint(border, (config.frame_size - config.peripheral_task_size) // 8,
                                        (batch_size, 1))
        periphery_pos = torch.hstack((periphery_pos_x, periphery_pos_y))

        if config.intent_size is not None:
            intent_pos_x = torch.randint(border, config.frame_size - config.intent_size, (batch_size, 1))
            intent_pos_y = torch.randint(int((config.frame_size - config.intent_size) / (9 / 8)),
                                         config.frame_size - config.intent_size - border, (batch_size, 1))
            intent_pos = torch.hstack((intent_pos_x, intent_pos_y))

    # swap in case train/test biased
    if mode == "test" and config.biased:
        periphery_pos_x = torch.randint(border, (config.frame_size - config.peripheral_task_size - border),
                                        (batch_size, 1))
        # limit y dim in the top part
        periphery_pos_y = torch.randint(int((config.frame_size - config.peripheral_task_size) / (9 / 8)),
                                        config.frame_size - config.peripheral_task_size - border,
                                        (batch_size, 1))
        periphery_pos = torch.hstack((periphery_pos_x, periphery_pos_y))
        if config.intent_size is not None:
            intent_pos_x = torch.randint(config.frame_size - config.intent_size - border, (batch_size, 1))
            intent_pos_y = torch.randint(border, (config.frame_size - config.intent_size) // 8,
                                         (batch_size, 1))
            intent_pos = torch.hstack((intent_pos_x, intent_pos_y))

    if config.intent_size is not None:
        intent_choice = torch.randint(0, 2, (1,))

    # generate both intents for this batch

    targets_batch_list = []

    # still first example
    i = 0
    image = periphery_input
    for j in range(2):
        background[j, :, periphery_pos[j, 1]:periphery_pos[j, 1] + config.peripheral_task_size,
        periphery_pos[j, 0]:periphery_pos[j, 0] + config.peripheral_task_size] = image

    background[0, :, intent_pos[0, 1]:intent_pos[0, 1] + config.intent_size,
    intent_pos[0, 0]:intent_pos[0, 0] + config.intent_size] = circle_intent
    targets_batch_list.append(middle_target.unsqueeze(dim=0))

    background[1, :, intent_pos[1, 1]:intent_pos[1, 1] + config.intent_size,
    intent_pos[1, 0]:intent_pos[1, 0] + config.intent_size] = square_intent
    targets_batch_list.append(periphery_target.unsqueeze(dim=0))

    # second example

    background[2, :, c_p_top:c_p_bottom, c_p_top:c_p_bottom] = periphery_input
    background[3, :, c_p_top:c_p_bottom, c_p_top:c_p_bottom] = periphery_input

    image = middle_input
    for i in range(2, 4):
        background[i, :, periphery_pos[i, 1]:periphery_pos[i, 1] + config.peripheral_task_size,
        periphery_pos[i, 0]:periphery_pos[i, 0] + config.peripheral_task_size] = image

    # target/intent positioning

    # selected periphery task
    background[2, :, intent_pos[2, 1]:intent_pos[2, 1] + config.intent_size,
    intent_pos[2, 0]:intent_pos[2, 0] + config.intent_size] = circle_intent
    targets_batch_list.append(periphery_target.unsqueeze(dim=0))

    background[3, :, intent_pos[3, 1]:intent_pos[3, 1] + config.intent_size,
    intent_pos[3, 0]:intent_pos[3, 0] + config.intent_size] = square_intent
    targets_batch_list.append(middle_target.unsqueeze(dim=0))

    targets_batch_list = torch.cat(targets_batch_list)

    # normalize the whole frame
    example_list.append(norm_transform(background))

    return torch.cat(example_list).numpy(), targets_batch_list.numpy(), None
