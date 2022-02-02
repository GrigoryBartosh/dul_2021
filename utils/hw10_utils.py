import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid

from .utils import *


def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])

    _ = MNIST(root='./', train=True, transform=transform, download=True)

    _ = MNIST(root='./', train=False, transform=transform, download=True)


def get_mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])

    train_set = MNIST(root='./', train=True, transform=transform, download=True)

    test_set = MNIST(root='./', train=False, transform=transform, download=True)

    return train_set, test_set


def tensor_to_image(x):
    x = x.detach().cpu().numpy()
    x = x.transpose(1, 2, 0)
    x = (x + 1) / 2
    return x


def show_samples(img):
    img = torch.Tensor(img)
    img = make_grid(img, nrow=10)
    img = tensor_to_image(img)
    img = img.clip(0, 1)

    plt.figure(figsize=(15, 15))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def plot_ce_training(train_ae, train_clf, title='Losses'):
    plt.figure()
    x = np.arange(len(train_ae)) / len(train_ae)
    x2 = np.arange(len(train_clf)) / len(train_clf) * len(train_ae)

    plt.plot(x, train_ae, label='ae loss')
    plt.plot(x2, train_clf, label='clf loss')

    plt.legend()
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')


def q1_results(q1):
    train_data, test_data = get_mnist()
    aeloss, clfloss, samples = q1(train_data, test_data)

    plot_ce_training(aeloss, clfloss)
    show_samples(samples)


def plot_rt_training(losses, title='Losses'):
    plt.figure()
    x = np.arange(len(losses[0]))

    plt.plot(x, losses[0], label='loss')

    plt.legend()
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    plt.figure()
    x = np.arange(len(losses[1]))

    plt.plot(x, losses[1], label='acc')

    plt.legend()
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.show()


def q2_results(q2):
    train_data, _ = get_mnist()
    loss, acc = q2(train_data)

    plot_rt_training((loss, acc))


