import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # rgb 平均值 方差
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ]
)

# 训练数据集
trainset = torchvision.datasets.CIFAR10(root="./data",
                                        train=True,
                                        download=True,
                                        transform=transform,
                                        )
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# 训练数据集
testset = torchvision.datasets.CIFAR10(root="./data",
                                       train=False,
                                       download=True,
                                       transform=transform,
                                       )
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

import matplotlib.pyplot as plt
import numpy as np


# %matplotlip inline
def imshow(img):
    # 输入数据
    img = img / 2 + 0.5
    nping = img.numpy()
    nping = np.transpose(nping, (1, 2, 0))  # [h,w,c]
    plt.imshow(nping)

dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))