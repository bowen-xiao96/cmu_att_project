import os, sys
import xml.etree.ElementTree as ET
import numpy as np

from PIL import Image

IMAGE_SIZE = 224


def _load_imagenet_notation(filename):
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


# crop images according to bounding boxes
def crop_image(anno_path, image_path, save_to, class_id, size_thresh=None, ratio_thresh=None):
    for f in os.listdir(os.path.join(anno_path, class_id)):
        anno_filename = os.path.join(anno_path, class_id, f)
        bboxes = _load_imagenet_notation(anno_filename)

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
            ratio = float(max(w, h)) / min(w, h)  # >= 1
            if ratio > ratio_thresh:
                continue

        # crop and resize image
        img = Image.open(os.path.join(image_path, class_id, os.path.splitext(f)[0] + '.JPEG')).convert('RGB')
        img = img.crop((xmin, ymin, xmax, ymax))
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.LANCZOS)

        if not os.path.exists(os.path.join(save_to, class_id)):
            os.makedirs(os.path.join(save_to, class_id))

        img.save(os.path.join(save_to, class_id, os.path.splitext(f)[0] + '.JPEG'))
