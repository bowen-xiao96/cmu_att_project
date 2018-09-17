import os, sys
import shutil
import numpy as np
import h5py

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from PIL import Image

test_root = r'/data2/bowenx/dataset/occlusion/occ'
train_root = r'/data2/bowenx/dataset/occlusion/SP_final'
image_root = r'/data2/bowenx/dataset/occlusion/imagenet_images'
save_to = r'/data2/bowenx/dataset/occlusion'

classes = ('aeroplane', 'bicycle', 'bus', 'car', 'motorbike', 'train')
levels = ('one', 'five', 'nine')

if __name__ == '__main__':
    # load testing set list
    test_set = {k: set() for k in classes}
    for c in classes:
        f = os.path.join(test_root, '%sLEVEL%s' % (c, levels[0].upper()))
        for ff in os.listdir(f):
            synset_id, img_id = ff.split('_')[:2]
            file_id = '%s_%s' % (synset_id, img_id)
            test_set[c].add(file_id)

    print('Test set:')
    print(sum([len(k) for k in test_set.values()]))

    # load full list
    full_set = {k: set() for k in classes}
    for c in classes:
        f = os.path.join(train_root, c + '_imagenet', 'transfered')
        for ff in os.listdir(f):
            file_id, _ = os.path.splitext(ff)
            full_set[c].add(file_id)

    print('Full set:')
    print(sum([len(k) for k in full_set.values()]))

    train_set = {k: full_set[k] - test_set[k] for k in classes}

    print('Train set:')
    print(sum([len(k) for k in train_set.values()]))

    # move files into training set and testing set directories
    # training set
    for c in classes:
        os.makedirs(os.path.join(save_to, 'train_set', c), exist_ok=True)
        os.makedirs(os.path.join(save_to, 'test_set', 'original', c), exist_ok=True)

        for level in levels:
            os.makedirs(os.path.join(save_to, 'test_set', level, c), exist_ok=True)

    # move original images
    for f in os.listdir(image_root):
        ff = os.path.join(image_root, f)
        for fff in os.listdir(ff):
            file_id, _ = os.path.splitext(fff)

            # find whether in training or testing set
            for c in classes:
                if file_id in train_set[c]:
                    shutil.copy(os.path.join(ff, fff), os.path.join(save_to, 'train_set', c, fff))
                    break

                elif file_id in test_set[c]:
                    shutil.copy(os.path.join(ff, fff), os.path.join(save_to, 'test_set', 'original', c, fff))
                    break

    # move occluded images
    for level in levels:
        for c in classes:
            f = os.path.join(test_root, '%sLEVEL%s' % (c, level.upper()))
            for ff in os.listdir(f):
                # read matlab file and save as jpg
                mat_file = h5py.File(os.path.join(f, ff), 'r')
                img = np.array(mat_file['record']['img'])
                img = np.transpose(img, (2, 1, 0))
                img = Image.fromarray(img, mode='RGB')

                file_id, _ = os.path.splitext(ff)
                img.save(os.path.join(save_to, 'test_set', level, c, file_id + '.JPEG'))


def get_dataloader(is_train, batch_size, num_workers, level=None):
    # get occlusion dataset

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean, std)

    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.LANCZOS),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        dataset = ImageFolder(
            os.path.join(save_to, 'train_set'),
            transform=transform
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            normalize
        ])

        dataset = ImageFolder(
            os.path.join(save_to, 'test_set', level),
            transform=transform
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False
        )

    return dataloader
