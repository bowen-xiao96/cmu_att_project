import os, sys
import pickle
import torch
from torchvision import transforms

from PIL import Image

ROOT_DOR = r'cub_cropped'
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.LANCZOS),
    transforms.ToTensor()
])

if __name__ == '__main__':
    all_images = list()

    # take training set only
    with open('cub_metadata.pkl', 'rb') as f_in:
        _, train_list, _ = pickle.load(f_in)

    # neglect class information
    for f, _, _ in train_list:
        image_filename = os.path.join(ROOT_DOR, f)
        img = Image.open(image_filename).convert('RGB')

        img_tensor = transform(img)
        all_images.append(img_tensor)

    all_images = torch.stack(all_images)
    print(all_images.size())

    # N x C x H x W
    all_images = torch.transpose(all_images, 0, 1).contiguous()
    all_images = all_images.view(all_images.size(0), -1)
    all_images = all_images.double()

    mean = torch.mean(all_images, dim=1)
    std = torch.std(all_images, dim=1)

    print(mean, std)
