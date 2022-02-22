from .utils import *

import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import make_grid


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


def show_samples(samples):
    img = torch.Tensor(samples)
    img = img.view(-1, 1, 28, 28)
    img = make_grid(img, nrow=10)
    img = tensor_to_image(img)
    img = img.clip(0, 1)

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def plot_q1_training_plot(train_con, train_reg, title='Losses'):
    plt.figure()
    x = np.arange(len(train_con))

    plt.plot(x, train_con, label='contr. loss')
    plt.plot(x, train_reg, label='reg. loss')

    plt.legend()
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()


def tensor_to_image(x):
    x = x.detach().cpu().numpy()
    x = x.transpose(1, 2, 0)
    return x


def q1_results(q1):
    train_data, _ = get_mnist()
    closs, rloss, samples = q1(train_data)

    plot_q1_training_plot(closs, rloss)
    show_samples(samples)


def load_smiley_face(n):
    count = n
    rand = np.random.RandomState(0)
    a = [[-1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    b = [[1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    c = np.c_[2 * np.cos(np.linspace(0, np.pi, count // 3)),
              -np.sin(np.linspace(0, np.pi, count // 3))]
    c += rand.randn(*c.shape) * 0.2
    data_x = np.concatenate([a, b, c], axis=0)
    data_y = np.array([0] * len(a) + [1] * len(b) + [2] * len(c))
    perm = rand.permutation(len(data_x))
    return data_x[perm], data_y[perm]


def q2_sample_data():
    train_data, train_labels = load_smiley_face(2000)
    test_data, test_labels = load_smiley_face(1000)
    return train_data, train_labels, test_data, test_labels


def plot_q2_training_plot(losses, title='Losses'):
    plt.figure()
    x = np.arange(len(losses))

    plt.plot(x, losses, label='loss')

    plt.legend()
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()


def q2_results(q2):
    train_data, train_labels, _, _ = q2_sample_data()
    loss, samples = q2(train_data, train_labels)

    plot_q2_training_plot(loss)
    samples = np.split(samples, 3, axis=0)
    for i, sample in enumerate(samples):
        plt.scatter(sample[:, 0], sample[:, 1], label=f'class = {i}')
    plt.show()


def plot_b_training_plot(losses_clf, losses_ebm, title='Losses'):
    plt.figure()
    x = np.arange(len(losses_clf))

    plt.plot(x, losses_clf, label='loss clf')
    plt.plot(x, losses_ebm, label='loss ebm')

    plt.legend()
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()


def b_results(b):
    train_data, train_labels, _, _ = q2_sample_data()
    loss_clf, loss_ebm, samples = b(train_data, train_labels)

    plot_b_training_plot(loss_clf, loss_ebm)
    samples = np.split(samples, 3, axis=0)
    for i, sample in enumerate(samples):
        plt.scatter(sample[:, 0], sample[:, 1], label=f'class = {i}')
    plt.show()
