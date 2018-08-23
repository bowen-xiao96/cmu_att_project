import os, sys
import pickle
import numpy as np
from PIL import Image

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])


class ImageNetDataset(Dataset):
    def __init__(self, metadata_file, tag, transform, labels=None):
        super(ImageNetDataset, self).__init__()

        with open(metadata_file, 'rb') as f_in:
            _, _, train_data, val_data = pickle.load(f_in)

        self.tag = tag
        self.transform = transform

        if tag == 'train':
            self.data = train_data
        else:
            self.data = val_data

        if labels:
            labels = set(labels)
            self.data = [(img_name, bboxes, cat) for img_name, bboxes, cat in self.data if cat in labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_name, bboxes, cat = self.data[item]
        img = Image.open(img_name).convert('RGB')

        transformed = self.transform(img, bboxes)
        if isinstance(transformed, (list, tuple)):
            return list(transformed) + [cat]
        else:
            return transformed, cat


def transform_with_bbox(img, bboxes):
    # generate a object map according to the bboxes
    # as well as transform the image into pytorch format

    raise NotImplementedError


def get_dataloader(metadata_file, batch_size, num_workers, labels=None):
    train_dataset = ImageNetDataset(
        metadata_file,
        'train',
        lambda img, bboxes: train_transform(img),
        labels=labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_dataset = ImageNetDataset(
        metadata_file,
        'test',
        lambda img, bboxes: test_transform(img),
        labels=labels
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )

    return train_loader, test_loader
