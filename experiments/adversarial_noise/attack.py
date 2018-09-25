import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# calculate bound for clipping
lower_bound = (0.0 - mean) / std
lower_bound = torch.from_numpy(
    np.reshape(lower_bound, (1, 3, 1, 1)).astype(np.float32)
)
lower_bound = lower_bound.cuda()

upper_bound = (1.0 - mean) / std
upper_bound = torch.from_numpy(
    np.reshape(upper_bound, (1, 3, 1, 1)).astype(np.float32)
)
upper_bound = upper_bound.cuda()


def FGSM_attack(model, img, y, eps, criterion=F.cross_entropy, iter_count=1, visualize=False):
    # assume `img` and `y` are pytorch Tensors
    # `model` is already cuda and eval, and all its parameters do not need gradient
    # `y` is standard image label
    # return image representation if `visualize` is True
    img = A.Variable(img.cuda(), requires_grad=True)
    y = A.Variable(y.cuda(), requires_grad=False)

    scores = list()  # only keep the first (unattacked) and last score

    for i in range(iter_count):
        # calculate noise
        if img.grad is not None:
            img.grad.data.zero_()

        # TODO: support for newloss model
        # low priority
        pred = model(img)

        if len(pred.size()) == 3:
            # only take the final prediction
            pred = pred[:, -1, :]

        if i == 0:
            scores.append(pred.data.cpu())

        loss = criterion(pred, y)
        loss.backward()

        # clip gradient
        n = eps * torch.sign(img.grad.data)

        # add to image and clamp
        new_image = img.data + n
        new_image = torch.max(torch.min(new_image, upper_bound), lower_bound)
        img.data = new_image

        if i == iter_count - 1:
            final_pred = model(img)
            if len(final_pred.size()) == 3:
                final_pred = final_pred[:, -1, :]

            scores.append(final_pred.data.cpu())
            scores = torch.stack(scores, dim=1)  # batch_size * 2 * num_class

            if visualize:
                return scores, img.data.cpu().numpy()
            else:
                return scores
