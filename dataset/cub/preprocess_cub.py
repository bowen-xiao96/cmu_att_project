import os, sys
import numpy as np
import gzip
import pickle

from PIL import Image
from scipy.io import loadmat

ROOT_DIR = '/data2/bowenx/dataset/fine-grained/cub-bird'


def _read_txt(filename):
    with open(filename, 'r') as f_in:
        lines = f_in.readlines()

    return [l.strip() for l in lines if l.strip()]


def load_metadata(root_dir, output_file):
    classes = _read_txt(os.path.join(root_dir, 'lists', 'classes.txt'))
    output = [classes]  # classes, train, test

    for tag in ('train', 'test'):
        data_list = list()
        image_list = _read_txt(os.path.join(root_dir, 'lists', tag + '.txt'))

        for f in image_list:
            class_tag, _ = f.split('/')
            class_id = classes.index(class_tag)

            # load bbox
            anno_file_name = os.path.join(root_dir, 'annotations', 'annotations-mat', os.path.splitext(f)[0] + '.mat')
            mat_file = loadmat(anno_file_name)
            bbox = np.array((
                mat_file['bbox']['left'].item().item(),
                mat_file['bbox']['top'].item().item(),
                mat_file['bbox']['right'].item().item(),
                mat_file['bbox']['bottom'].item().item(),
            ))

            assert os.path.exists(os.path.join(root_dir, 'images', f))
            data_list.append((class_id, f, bbox))

        output.append(data_list)

    with gzip.open(output_file, 'wb') as f_out:
        pickle.dump(tuple(output), f_out, pickle.HIGHEST_PROTOCOL)


def crop_images(metadata_file, root_dir, save_to):
    with gzip.open(metadata_file, 'rb') as f_in:
        _, train_list, test_list = pickle.load(f_in)

    data = train_list + test_list
    for class_id, img_filename, bbox in data:
        img = Image.open(os.path.join(root_dir, 'images', img_filename)).convert('RGB')

        cropped = img.crop(bbox)
        output_filename = os.path.join(save_to, img_filename)
        if not os.path.exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename))

        cropped.save(output_filename)


if __name__ == '__main__':
    load_metadata(ROOT_DIR, 'cub_metadata.pkgz')
    crop_images('cub_metadata.pkgz', ROOT_DIR, 'cropped')
