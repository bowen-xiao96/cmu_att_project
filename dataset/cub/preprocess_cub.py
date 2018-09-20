import os, sys
import numpy as np
import pickle

from PIL import Image
from scipy.io import loadmat

ROOT_DIR = '/data2/bowenx/dataset/fine-grained/cub-bird'


def _read_txt(filename):
    with open(filename, 'r') as f_in:
        lines = f_in.readlines()

    return [l.strip() for l in lines if l.strip()]


# crop CUB images according to their bounding boxes
# then dump image metadata to a pickle file
def preprocess(root_dir, img_output_dir):
    classes = _read_txt(os.path.join(root_dir, 'lists', 'classes.txt'))
    train_list = list()
    test_list = list()

    for flag in ('train', 'test'):
        image_list = _read_txt(os.path.join(root_dir, 'lists', flag + '.txt'))

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

            # check file existence and record it
            image_filename = os.path.join(root_dir, 'images', f)
            assert os.path.exists(image_filename)
            locals()[flag + '_list'].append((f, class_id, bbox))

            # crop the image according to the bounding box
            img = Image.open(image_filename).convert('RGB')

            cropped = img.crop(bbox)
            output_filename = os.path.join(img_output_dir, f)
            if not os.path.exists(os.path.dirname(output_filename)):
                os.makedirs(os.path.dirname(output_filename))

            cropped.save(output_filename)

    print(len(train_list))
    print(len(test_list))
    with open('cub_metadata.pkl', 'wb') as f_out:
        pickle.dump((classes, train_list, test_list), f_out, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    preprocess(ROOT_DIR, 'cub_cropped')
