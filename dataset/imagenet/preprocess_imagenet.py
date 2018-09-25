# read bounding box annotation for each image
# and dump them in a pickle file

# also provide tool to crop image according to the bbox

import os, sys
import pickle
import xml.etree.ElementTree as ET
import numpy as np

from PIL import Image

image_size = 224


def load_imagenet_notation(filename, label_dict):
    # given bbox xml file name
    # returns (n_bbox, 5) numpy array
    # (xmin, ymin, xmax, ymax, cat)

    tree = ET.parse(filename)
    root = tree.getroot()
    bboxes = list()

    for object_node in root.findall('object'):
        bbox = object_node.find('bndbox')

        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        cat = label_dict[object_node.find('name').text]

        bboxes.append(np.array([xmin, ymin, xmax, ymax, cat], dtype=np.int64))

    return np.stack(bboxes)


def dump_dataset(anno_path, image_path, label_file, val_set_label_file):
    # load all the image names and bounding boxes
    # save in a pickle file

    # parse labels
    labels = list()
    label_dict = dict()

    with open(label_file, 'r') as f_in:
        lines = [l.strip() for l in f_in.readlines() if l.strip()]

    for l in lines:
        class_id, cat, description = l.split()
        # in annotation file, category number starts from 1
        cat = int(cat) - 1
        assert cat == len(labels)
        labels.append((class_id, description))
        label_dict[class_id] = cat

    # load training set
    train_data = list()
    for cat, (class_id, _) in enumerate(labels):
        anno_dir = os.path.join(anno_path, 'train', class_id)
        for anno_file in os.listdir(anno_dir):
            bboxes = load_imagenet_notation(os.path.join(anno_dir, anno_file), label_dict)
            img_name = os.path.join(image_path, 'train', class_id, os.path.splitext(anno_file)[0] + '.JPEG')
            assert os.path.exists(img_name)

            train_data.append((img_name, bboxes, cat))

    # load val set
    val_data = list()
    with open(val_set_label_file, 'r') as f_in:
        lines = [int(l.strip()) for l in f_in.readlines() if l.strip()]

    # altogether 50000 val images
    assert len(lines) == 50000

    for i, cat in enumerate(lines):
        anno_file_name = os.path.join(anno_path, 'val', 'ILSVRC2012_val_%08d.xml' % (i + 1))
        bboxes = load_imagenet_notation(anno_file_name, label_dict)
        img_name = os.path.join(image_path, 'val', 'ILSVRC2012_val_%08d.JPEG' % (i + 1))
        assert os.path.exists(img_name)

        # cat start from 1
        val_data.append((img_name, bboxes, cat - 1))

    with open('imagenet_metadata.pkl', 'wb') as f_out:
        pickle.dump((labels, train_data, val_data), f_out, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    dump_dataset(
        '/data2/leelab/ILSVRC2015_CLS-LOC/ILSVRC2015/Annotations/CLS-LOC',
        '/data2/leelab/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC',
        'map_clsloc.txt',
        'ILSVRC2014_clsloc_validation_ground_truth.txt'
    )
