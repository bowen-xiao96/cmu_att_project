import os, sys
import pickle
import xml.etree.ElementTree as ET
import numpy as np

from PIL import Image

IMAGE_SIZE = 224


def load_imagenet_notation(filename):
    # given bbox xml file name
    # returns (n_bbox, 4) numpy array

    tree = ET.parse(filename)
    root = tree.getroot()
    bboxes = list()

    for object_node in root.findall('object'):
        bbox = object_node.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        bboxes.append(np.array([xmin, ymin, xmax, ymax]))

    return np.stack(bboxes)


def crop_image(anno_path, image_path, save_to, class_id, size_thresh=None, ratio_thresh=None):
    # crop images according to bounding boxes
    # applies to a whole class

    for f in os.listdir(os.path.join(anno_path, class_id)):
        bboxes = load_imagenet_notation(os.path.join(anno_path, class_id, f))

        # throw away images with more than one bounding boxes
        if bboxes.shape[0] > 1:
            continue

        xmin, ymin, xmax, ymax = bboxes[0]
        w = xmax - xmin
        h = ymax - ymin
        if size_thresh:
            if min(w, h) < size_thresh:
                continue

        if ratio_thresh:
            ratio = float(min(w, h)) / max(w, h)  # <= 1
            if ratio < ratio_thresh:
                continue

        # crop and resize image
        img = Image.open(os.path.join(image_path, class_id, os.path.splitext(f)[0] + '.JPEG')).convert('RGB')
        img = img.crop((xmin, ymin, xmax, ymax))
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.LANCZOS)

        output_path = os.path.join(save_to, class_id)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        img.save(os.path.join(output_path, os.path.splitext(f)[0] + '.JPEG'))


def dump_dataset(anno_path, image_path, label_file):
    # load all the image names and bounding boxes
    # save in a pickle file

    flag = ('train', 'val')

    # parse labels
    labels = list()
    with open(label_file, 'r') as f_in:
        lines = f_in.readlines()
        for l in lines:
            l = l.strip()
            if not l: continue

            class_id, cat, description = l.split()
            assert int(cat) == len(labels) + 1
            labels.append((class_id, description))

    assert len(labels) == 1000

    for f in flag:
        # (filename, bboxes, cat)
        data = list()
        for cat, (class_id, _) in enumerate(labels):
            anno_dir = os.path.join(anno_path, f, class_id)
            for anno_file in os.listdir(anno_dir):
                bboxes = load_imagenet_notation(os.path.join(anno_dir, anno_file))
                img_name = os.path.join(image_path, f, class_id, os.path.splitext(anno_file)[0] + '.JPEG')
                assert os.path.exists(img_name)

                data.append((img_name, bboxes, cat))

        with open('imagenet_%s.pkl' % f, 'wb') as f_out:
            pickle.dump((data, labels), f_out, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    dump_dataset(
        '/data2/leelab/ILSVRC2015_CLS-LOC/ILSVRC2015/Annotations/CLS-LOC',
        '/data2/simingy/data/Imagenet',
        '/data2/bowenx/dataset/map_clsloc.txt'
    )
