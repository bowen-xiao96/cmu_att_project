import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A

mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))

lower_bound = torch.from_numpy((0.0 - mean) / std).float().cuda()
upper_bound = torch.from_numpy((1.0 - mean) / std).float().cuda()


def FGSM_attack(model, img, y, iter_count, eps):
    # img is pytorch cpu tensor
    # y is also cpu tensor (label for the image)

    original_img = img.clone().numpy()
    img = A.Variable(img.cuda(), requires_grad=True)
    y = A.Variable(y.cuda(), requires_grad=False)

    noise = list()
    img_with_noise = list()
    score = [model(img).data.cpu().numpy()]

    for i in range(iter_count):
        # calculate noise
        if img.grad is not None:
            img.grad.zero_()

        pred = model(img)
        if len(pred.size()) == 3:
            # only take the final prediction
            pred = pred[:, -1, :]

        loss = F.cross_entropy(pred, y)
        loss.backward()

        # clip gradient
        n = eps * torch.sign(img.grad.data)
        noise.append(n.cpu().numpy())

        # add to image and clamp
        new_image = img.data + n
        new_image = torch.max(torch.min(new_image, upper_bound), lower_bound)
        img_with_noise.append(new_image.cpu().numpy())

        img.data = new_image
        score.append(model(img).data.cpu().numpy())

    return original_img, noise, img_with_noise, score
