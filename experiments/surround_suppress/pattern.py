import numpy as np
from PIL import Image

import imagen
import matplotlib.pyplot as plt

mean_color = np.array([0.485, 0.456, 0.406])


def sinewave_pattern(size, freq, orientation, smoothing=0.003, img_size=224):
    region_scale = float(size) / img_size

    pattern_gen = imagen.SineGrating(
        orientation=orientation,
        scale=0.8,  # brightness
        frequency=freq,
        x=0.0,
        y=0.0,
        mask_shape=imagen.Disk(size=region_scale, smoothing=smoothing)
    )

    img = pattern_gen()
    img = Image.fromarray((img * 255.0).astype(np.uint8), mode='L')
    img = img.resize((img_size, img_size), resample=Image.LANCZOS)
    return img


if __name__ == '__main__':
    size = 72
    orientation = np.pi / 6.0

    pattern = sinewave_pattern(size, 50, orientation)
    plt.clf()
    plt.imshow(pattern)
    plt.show()
