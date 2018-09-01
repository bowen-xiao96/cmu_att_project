import os, sys
import random
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image

assert len(sys.argv) > 2
sample_count = int(sys.argv[1])
sigmas = [float(v) for v in sys.argv[2:]]

image_path = r'/data2/leelab/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/val'
output_path = r'/data2/simingy/data/noise_visualize/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

samples = random.sample(os.listdir(image_path), sample_count)

for f in samples:
    original_img = Image.open(os.path.join(image_path, f)).convert('RGB')
    img = np.array(original_img).astype(np.float32)

    plt.clf()
    col_count = len(sigmas) + 1
    for i in range(col_count):
        plt.subplot(1, col_count, i + 1)
        plt.axis('off')
        if i == 0:
            plt.imshow(original_img)
            plt.title(str(0.0))
        else:
            # add noise to the image
            n = np.random.normal(0.0, sigmas[i - 1], img.shape)
            noise_img = img + n
            noise_img = np.clip(noise_img, 0.0, 255.0).astype(np.uint8)
            noise_img = Image.fromarray(noise_img, mode='RGB')

            plt.imshow(noise_img)
            plt.title(str(sigmas[i - 1]))

    plt.savefig(os.path.join(output_path, f), dpi=1000)
