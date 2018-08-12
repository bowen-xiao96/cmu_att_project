import os, sys
import numpy as np
import pickle

import torch
import torch.nn.functional as F

from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def _read_txt(filename):
    with open(filename, 'r') as f_in:
        lines = f_in.readlines()

    return [l.strip() for l in lines if l.strip()]


# root_dir = '/data2/bowenx/attention/fine-grained/cub-bird'
def get_metadata(root_dir, metadata_file):
    classes = _read_txt(os.path.join(root_dir, 'lists', 'classes.txt'))
    image_list = {
        'train': _read_txt(os.path.join(root_dir, 'lists', 'train.txt')),
        'test': _read_txt(os.path.join(root_dir, 'lists', 'test.txt'))
    }

    # train and test
    output_metadata = {
        'train': list(),
        'test': list()
    }

    for flag in ('train', 'test'):
        for f in image_list[flag]:
            class_tag, _ = f.split('/')
            class_id = classes.index(class_tag)

            # load bbox
            anno_file_name = os.path.join(root_dir,
                                          'annotations',
                                          'annotations-mat',
                                          os.path.splitext(f)[0] + '.mat')

            mat_file = loadmat(anno_file_name)
            bbox = np.array((
                mat_file['bbox']['left'].item().item(),
                mat_file['bbox']['top'].item().item(),
                mat_file['bbox']['right'].item().item(),
                mat_file['bbox']['bottom'].item().item(),
            ))

            output_metadata[flag].append(
                (class_id, f, bbox)
            )

    with open(metadata_file, 'wb') as f_out:
        pickle.dump(output_metadata, f_out, pickle.HIGHEST_PROTOCOL)


# image_path = '/data2/bowenx/attention/fine-grained/cub-bird/images'
def crop(metadata_file, image_path, save_path):
    with open(metadata_file, 'rb') as f_in:
        data = pickle.load(f_in)

    data_list = data['train'] + data['test']
    for _, img_name, bbox in data_list:
        img = Image.open(os.path.join(image_path, img_name)).convert('RGB')

        cropped = img.crop(bbox)
        save_name = os.path.join(save_path, img_name)
        if not os.path.exists(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))

        cropped.save(save_name)


def calculate_mean_std(metadata_file, image_path, mean_std_file):
    with open(metadata_file, 'rb') as f_in:
        data = pickle.load(f_in)

    data_list = data['train']
    transform = transforms.ToTensor()
    imgs = list()

    for _, img_name, _ in data_list:
        img = Image.open(os.path.join(image_path, img_name)).convert('RGB')
        img = transform(img)

        # PIL tensor (C x H x W)
        imgs.append(img.view(img.size(0), -1))

    imgs = torch.cat(imgs, dim=1).cuda()

    mean = torch.mean(imgs, dim=1).cpu().numpy()
    std = torch.std(imgs, dim=1).cpu().numpy()

    np.savez(mean_std_file, mean=mean, std=std)


class CUBDataset(Dataset):
    def __init__(self, data_list, image_path, transform):
        self.data_list = data_list
        self.image_path = image_path
        self.transform = transform

        self.image_count = len(self.data_list)

    def __len__(self):
        return self.image_path

    def __getitem__(self, item):
        class_id, f, _ = self.data_list[item]

        img = Image.open(
            os.path.join(self.image_path, f)
        ).convert('RGB')

        img = self.transform(img)

        return img, class_id


def get_dataloaders(metadata_file, image_path, batch_size, num_workers):
    # creating dataset and dataloader
    mean = np.array([0.47350648, 0.47355667, 0.41435933])
    std = np.array([0.24078621, 0.23705634, 0.26397347])
    normalize = transforms.Normalize(mean, std)

    with open(metadata_file, 'rb') as f_in:
        data = pickle.load(f_in)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = CUBDataset(
        data['train'],
        image_path,
        train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    test_dataset = CUBDataset(
        data['test'],
        image_path,
        test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    return train_loader, test_loader
