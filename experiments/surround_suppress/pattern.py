import numpy as np
from PIL import Image

import imagen

bg_color = np.array([0.485, 0.456, 0.406])
bg_color = bg_color[np.newaxis, np.newaxis, :]
fg_color = np.array([0.0, 0.0, 0.0])
fg_color = fg_color[np.newaxis, np.newaxis, :]


def sinewave_pattern(size, freq, orientation, smoothing=0.003, img_size=224):
    region_scale = float(size) / img_size

    pattern_gen = imagen.SineGrating(
        orientation=orientation,
        scale=1.0,  # brightness
        frequency=freq,
        x=0.0,
        y=0.0,
        mask_shape=imagen.Disk(size=region_scale, smoothing=smoothing)
    )

    mask = pattern_gen()

    mask = mask[..., np.newaxis]
    img = mask * fg_color + (1 - mask) * bg_color
    img = Image.fromarray((img * 255.0).astype(np.uint8), mode='RGB')
    img = img.resize((img_size, img_size), resample=Image.LANCZOS)
    return img


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    size = 60
    orientation = np.pi / 6.0

    pattern = sinewave_pattern(size, 35, orientation)
    plt.clf()
    plt.imshow(pattern)
    plt.show()
