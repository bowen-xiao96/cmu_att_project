import os, sys
import random
import numpy as np
from PIL import Image

assert len(sys.argv) > 2
sample_count = int(sys.argv[1])
sigmas = [float(v) for v in sys.argv[2:]]

image_path = r'/data2/leelab/ILSVRC2015_CLS-LOC/ILSVRC2015/Data/CLS-LOC/val'
output_path = r'noise_visualize'

samples = random.sample(os.listdir(image_path), sample_count)

for f in samples:
    original_img = Image.open(os.path.join(image_path, f)).convert('RGB')
    img = np.array(original_img).astype(np.float32)

    output_dir = os.path.join(output_path, os.path.splitext(f)[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        for sigma in sigmas:
            if sigma > 0.0:
                n = np.random.normal(0.0, sigma, img.shape)
                noise_img = img + n
                noise_img = np.clip(noise_img, 0.0, 255.0).astype(np.uint8)
                noise_img = Image.fromarray(noise_img, mode='RGB')
            else:
                noise_img = Image.fromarray(img, mode='RGB')

            noise_img.save(os.path.join(output_dir, '%d.jpg' % int(sigma)))
