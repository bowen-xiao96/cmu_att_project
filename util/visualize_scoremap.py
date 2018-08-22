import numpy as np
from matplotlib import cm

from PIL import Image


def _get_jet():
    colormap_int = np.zeros((256, 3), np.uint8)

    for i in range(256):
        for j in range(3):
            colormap_int[i, j] = np.uint8(np.round(cm.jet(i)[j] * 255.0))

    return colormap_int


def _gray2color(gray_array, color_map):
    rows, cols = gray_array.shape
    color_array = np.zeros((rows, cols, 3), dtype=np.uint8)

    for i in range(0, rows):
        for j in range(0, cols):
            color_array[i, j, :] = color_map[gray_array[i, j]]

    return color_array


def log_transform(score_map):
    # apply log transform to the score map

    log_map = np.log(score_map)
    log_map -= np.min(log_map)
    log_map /= np.max(log_map)

    return log_map


def draw_image(original_image, score_maps, transform=None, size=224, padding=20):
    # original_image: batch_size * size * size * 3
    # score_maps: batch_size * unroll_count * size * size
    # return a list of batch_size PIL images

    batch_size, unroll_count = score_maps.shape[:2]
    color_map = _get_jet()

    if transform:
        score_maps = transform(score_maps)

    canvas = np.zeros((
        size * batch_size + padding * (batch_size - 1),
        size * (unroll_count + 1) + padding * unroll_count,
        3
    ), dtype=np.uint8)

    for i in range(batch_size):
        canvas[(size + padding) * i: size * (i + 1) + padding * i, :size, :] = original_image[i]

        for j in range(unroll_count):
            colored_map = _gray2color((255.0 * score_maps[i, j]).astype(np.uint8), color_map)
            canvas[(size + padding) * i: size * (i + 1) + padding * i,
            (size + padding) * (j + 1):size * (j + 2) + padding * (j + 1), :] = colored_map

    return Image.fromarray(canvas, mode='RGB')
