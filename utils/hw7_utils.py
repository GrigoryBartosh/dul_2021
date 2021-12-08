import cv2
import scipy.ndimage
import torch.nn as nn
import torch.utils.data
import torchvision
from PIL import Image as PILImage
from torchvision import transforms as transforms

from .hw7_is.hw7_models import GoogLeNet
from .pytorch_utils import *
from .utils import *

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import numpy as np
import math
import sys

softmax = None
model = None
device = torch.device("cuda:0")


def plot_gan_training(losses, title, fname):
    plt.figure()
    n_itr = len(losses)
    xs = np.arange(n_itr)

    plt.plot(xs, losses, label='loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss')
    savefig(fname)


def q1_gan_plot(data, samples, xs, ys, title, fname):
    plt.figure()
    plt.hist(samples, bins=50, density=True, alpha=0.7, label='fake')
    plt.hist(data, bins=50, density=True, alpha=0.7, label='real')

    plt.plot(xs, ys, label='discrim')
    plt.legend()
    plt.title(title)
    savefig(fname)


def calculate_is(samples):
    assert (type(samples[0]) == np.ndarray)
    assert (len(samples[0].shape) == 3)

    model = GoogLeNet().to(device)
    model.load_state_dict(torch.load("dul_2021/utils/hw7_is/classifier.pt"))
    softmax = nn.Sequential(model, nn.Softmax(dim=1))

    bs = 100
    softmax.eval()
    with torch.no_grad():
        preds = []
        n_batches = int(math.ceil(float(len(samples)) / float(bs)))
        for i in range(n_batches):
            sys.stdout.write(".")
            sys.stdout.flush()
            inp = FloatTensor(samples[(i * bs):min((i + 1) * bs, len(samples))]).to(device)
            pred = get_numpy(softmax(inp))
            preds.append(pred)
    preds = np.concatenate(preds, 0)
    kl = preds * (np.log(preds) - np.log(np.expand_dims(np.mean(preds, 0), 0)))
    kl = np.mean(np.sum(kl, 1))
    return np.exp(kl)


def load_q1_data():
    train_data = torchvision.datasets.CIFAR10("./data", transform=torchvision.transforms.ToTensor(),
                                              download=True, train=True)
    return train_data


def visualize_q1_data():
    train_data = load_q1_data()
    imgs = train_data.data[:100]
    show_samples(imgs, title=f'CIFAR-10 Samples')


def q1_save_results(fn):
    train_data = load_q1_data()
    train_data = train_data.data.transpose((0, 3, 1, 2)) / 255.0
    train_losses, samples = fn(train_data)

    print("Inception score:", calculate_is(samples.transpose([0, 3, 1, 2])))
    plot_gan_training(train_losses, 'Q1 Losses', 'results/q1_losses.png')
    show_samples(samples[:100] * 255.0, fname='results/q1_samples.png', title=f'CIFAR-10 generated samples')


######################
##### Question 2 #####
######################

def load_q2_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return train_data, test_data


def visualize_q2_data():
    train_data, _ = load_q2_data()
    imgs = train_data.data[:100]
    show_samples(imgs.reshape([100, 28, 28, 1]) * 255.0, title=f'MNIST samples')


def plot_q2_supervised(pretrained_losses, random_losses, title, fname):
    plt.figure()
    xs = np.arange(len(pretrained_losses))
    plt.plot(xs, pretrained_losses, label='bigan')
    xs = np.arange(len(random_losses))
    plt.plot(xs, random_losses, label='random init')
    plt.legend()
    plt.title(title)
    savefig(fname)


def q2_save_results(fn):
    train_data, test_data = load_q2_data()
    gan_losses, samples, reconstructions, pretrained_losses, random_losses = fn(train_data, test_data)

    plot_gan_training(gan_losses, 'Q2 Losses', 'results/q2_gan_losses.png')
    plot_q2_supervised(pretrained_losses, random_losses, 'Linear classification losses',
                       'results/q2_supervised_losses.png')
    show_samples(samples * 255.0, fname='results/q2_samples.png', title='BiGAN generated samples')
    show_samples(reconstructions * 255.0, nrow=20, fname='results/q3_reconstructions.png',
                 title=f'BiGAN reconstructions')
    print('BiGAN final linear classification loss:', pretrained_losses[-1])
    print('Random encoder linear classification loss:', random_losses[-1])


def get_colored_mnist(data):
    # from https://www.wouterbulten.nl/blog/tech/getting-started-with-gans-2-colorful-mnist/
    # Read Lena image
    lena = PILImage.open('dul_2021/utils/hw7_is/lena.jpg')

    # Resize
    batch_resized = np.asarray([scipy.ndimage.zoom(image, (2.3, 2.3, 1), order=1) for image in data])

    # Extend to RGB
    batch_rgb = np.concatenate([batch_resized, batch_resized, batch_resized], axis=3)

    # Make binary
    batch_binary = (batch_rgb > 0.5)

    batch = np.zeros((data.shape[0], 28, 28, 3))

    for i in range(data.shape[0]):
        # Take a random crop of the Lena image (background)
        x_c = np.random.randint(0, lena.size[0] - 64)
        y_c = np.random.randint(0, lena.size[1] - 64)
        image = lena.crop((x_c, y_c, x_c + 64, y_c + 64))
        image = np.asarray(image) / 255.0

        # Invert the colors at the location of the number
        image[batch_binary[i]] = 1 - image[batch_binary[i]]

        batch[i] = cv2.resize(image, (0, 0), fx=28 / 64, fy=28 / 64, interpolation=cv2.INTER_AREA)
    return batch.transpose(0, 3, 1, 2)


def load_b5_data():
    train, _ = load_q2_data()
    mnist = np.array(train.data.reshape(-1, 28, 28, 1) / 255.0)
    colored_mnist = get_colored_mnist(mnist)
    return mnist.transpose(0, 3, 1, 2), colored_mnist


def visualize_cyclegan_datasets():
    mnist, colored_mnist = load_b5_data()
    mnist, colored_mnist = mnist[:100], colored_mnist[:100]
    show_samples(mnist.reshape([100, 28, 28, 1]) * 255.0, title=f'MNIST samples')
    show_samples(colored_mnist.transpose([0, 2, 3, 1]) * 255.0, title=f'Colored MNIST samples')


def b5_save_results(fn):
    mnist, cmnist = load_b5_data()

    m1, c1, m2, c2, m3, c3 = fn(mnist, cmnist)
    m1, m2, m3 = m1.repeat(3, axis=3), m2.repeat(3, axis=3), m3.repeat(3, axis=3)
    mnist_reconstructions = np.concatenate([m1, c1, m2], axis=0)
    colored_mnist_reconstructions = np.concatenate([c2, m3, c3], axis=0)

    show_samples(mnist_reconstructions * 255.0, nrow=20,
                 fname='figures/b5_mnist.png',
                 title=f'Source domain: MNIST')
    show_samples(colored_mnist_reconstructions * 255.0, nrow=20,
                 fname='figures/b5_colored_mnist.png',
                 title=f'Source domain: Colored MNIST')
    pass
