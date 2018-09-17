# download all image packages in the full set
# and then extracts them into output dir

import os, sys
import numpy as np

import urllib.request
import urllib.error

username = r'yimengzh'
key = r'3b9463ae4e1f2cd956dd417af2750b43c4481696'

root_dir = r'/data2/bowenx/dataset/occlusion/SP_final'
output_dir = r'imagenet_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
os.chdir(output_dir)

# get synset ids
synsets = set()
for f in os.listdir(root_dir):
    if f.endswith('.txt'):
        with open(os.path.join(root_dir, f), 'r') as f_in:
            for line in f_in.readlines():
                line = line.strip()
                if not line:
                    continue

                image_id, _ = line.split()
                syn_id, _ = image_id.split('_')
                synsets.add(syn_id)

print(len(synsets))
print(synsets)

for synset in synsets:
    url = r'http://www.image-net.org/download/' \
          r'synset?wnid=%s&username=%s&accesskey=%s&release=latest&src=stanford' % (synset, username, key)

    save_filename = synset + '.tar'
    urllib.request.urlretrieve(url, save_filename)

    # extract from the archive and delete it
    os.makedirs(synset)
    os.system('tar xf %s -C %s' % (save_filename, synset))
    os.remove(save_filename)
