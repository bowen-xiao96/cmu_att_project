import os, sys
import pickle
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from PIL import Image


class CUBDataset(Dataset):
    def __init__(self, data_list, image_path, transform):
        super(CUBDataset, self).__init__()

        self.data_list = data_list
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        f, class_id, _ = self.data_list[item]

        img = Image.open(os.path.join(self.image_path, f)).convert('RGB')
        img = self.transform(img)

        return img, class_id


def get_dataloader(metadata_file, image_path, batch_size, num_workers):
    with open(metadata_file, 'rb') as f_in:
        _, train_list, test_list = pickle.load(f_in)

    mean = np.array([0.4703, 0.4717, 0.4085])
    std = np.array([0.2409, 0.2381, 0.2648])
    normalize = transforms.Normalize(mean, std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=Image.LANCZOS),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = CUBDataset(train_list, image_path, train_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=Image.LANCZOS),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    test_dataset = CUBDataset(test_list, image_path, test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )

    return train_loader, test_loader
