import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from PIL import Image


def get_jet():
    colormap_int = np.zeros((256, 3), np.uint8)

    for i in range(0, 256, 1):
        colormap_int[i, 0] = np.uint8(np.round(cm.jet(i)[0] * 255.0))
        colormap_int[i, 1] = np.uint8(np.round(cm.jet(i)[1] * 255.0))
        colormap_int[i, 2] = np.uint8(np.round(cm.jet(i)[2] * 255.0))

    return colormap_int


def gray2color(gray_array, color_map):
    rows, cols = gray_array.shape
    color_array = np.zeros((rows, cols, 3), dtype=np.uint8)

    for i in range(0, rows):
        for j in range(0, cols):
            color_array[i, j] = color_map[gray_array[i, j]]

    color_image = Image.fromarray(color_array)
    return color_image


if __name__ == '__main__':
    # size = 32
    # padding = 10
    # color_map = get_jet()
    # save_dir = 'score_map_visualize'
    #
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #
    # for flag in ('train', 'test'):
    #     data = np.load(flag + '.npz')
    #     images, score_maps = data['images'], data['score_maps']
    #
    #     for i in range(images.shape[0]):
    #         image = images[i]
    #
    #         output_img = np.zeros((size, size * 4 + padding * 3, 3), dtype=np.uint8)
    #         output_img[:, :size, :] = image
    #
    #         for j in range(3):
    #             output_img[:, (size + padding) * (j + 1):size * (j + 2) + padding * (j + 1), :] = \
    #                 gray2color((255.0 * score_maps[i, j]).astype(np.uint8), color_map)
    #
    #         Image.fromarray(output_img, mode='RGB').save(os.path.join(save_dir, '%s_%d.png' % (flag, i)))

    size = 224
    padding = 20
    color_map = get_jet()
    save_dir = 'imagenet_visualize'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = np.load('imagenet.npz')
    images, score_maps = data['images'], data['score_maps']
    score_maps = np.minimum(score_maps * 500, 1.0)

    for i in range(images.shape[0]):
        image = images[i]

        output_img = np.zeros((size, size * 4 + padding * 3, 3), dtype=np.uint8)
        output_img[:, :size, :] = image

        for j in range(3):
            output_img[:, (size + padding) * (j + 1):size * (j + 2) + padding * (j + 1), :] = \
                gray2color((255.0 * score_maps[i, j]).astype(np.uint8), color_map)

        Image.fromarray(output_img, mode='RGB').save(os.path.join(save_dir, '%d.png' % i))
