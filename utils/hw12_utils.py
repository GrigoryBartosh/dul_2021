from .utils import *


class SSDataset:
    def __init__(self, data, c=4000, train=True):
        self.data = data
        self.c = c
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x, y = self.data[i]
        if not self.train:
            return x, y

        y = y if i < self.c else -1
        return x, y


def q12_results(q):
    train_data, test_data = get_data('CIFAR10')
    train_data, test_data = SSDataset(train_data), SSDataset(test_data)

    losses, accs = q(train_data, test_data)

    plot_training(losses, 'Loss')
    plot_training(accs, 'Accuracy')


def b_resul(b):
    train_data, test_data = get_data('MNIST', binary=True)
    train_data, test_data = SSDataset(train_data), SSDataset(test_data)
    losses, accs = b(train_data, test_data)

    plot_training(losses, 'Loss')
    plot_training(accs, 'Accuracy')
