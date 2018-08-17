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


def draw_image(original_image, score_maps, size=224, padding=20):
    # original_image: batch_size * size * size * 3
    # score_maps: batch_size * unroll_count * size * size
    # return a list of batch_size PIL images

    batch_size, unroll_count = score_maps.shape[:2]
    output_imgs = list()
    color_map = get_jet()

    score_maps = np.log(score_maps)
    score_maps -= np.min(score_maps)
    score_maps /= np.max(score_maps)

    for i in range(batch_size):
        canvas = np.zeros((size, size * (unroll_count + 1) + padding * unroll_count, 3), dtype=np.uint8)
        canvas[:, :size, :] = original_image[i]

        for j in range(unroll_count):
            colored_map = gray2color((255.0 * score_maps[i, j]).astype(np.uint8), color_map)
            canvas[:, (size + padding) * (j + 1):size * (j + 2) + padding * (j + 1), :] = colored_map

        output_imgs.append(Image.fromarray(canvas, mode='RGB'))

    return output_imgs
