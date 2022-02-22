import torch.nn as nn
import torch.optim as opt
import torch.utils.data as data
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
import torch

from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

from .utils import *


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(128, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 10))

    def forward(self, x):
        return self.model(x)

    def decision_function_loader(self, dataloader):
        y_pred = []
        y_true = []

        for (x, y) in dataloader:
            res = self(x.float())
            res = torch.argmax(res, dim=-1)
            y_pred.append(res.squeeze().detach().cpu().numpy())
            y_true.append(y.squeeze().detach().cpu().numpy())

        return np.hstack(y_pred), np.hstack(y_true)

    def fit(self, traindata, lr=1e-3, bs=100, num_epochs=100):
        optim = opt.Adam(self.parameters(), lr=lr)
        scheduler = opt.lr_scheduler.ExponentialLR(optim, gamma=0.99)

        data_loader = torch.utils.data.DataLoader(traindata, bs, shuffle=True)

        for i in range(num_epochs):
            running_loss = 0
            for (x, y) in data_loader:
                p = self(x)
                loss = F.cross_entropy(p, y)
                optim.zero_grad()
                loss.backward()
                optim.step()

                running_loss += loss.item()

            scheduler.step()


def test_classification(test_data, encoder, it=10):
    encoded = []
    ys = []
    test_loader = data.DataLoader(test_data, batch_size=100, shuffle=False)

    with torch.no_grad():
        for (x, y) in test_loader:
            encoded.append(encoder(x).cpu())
            ys.append(y)

    res = []
    for i in range(it):
        clf_train_data = torch.cat(encoded[:10 * i] + encoded[10 * (i + 1):], dim=0)
        clf_train_labels = torch.cat(ys[:10 * i] + ys[10 * (i + 1):])
        clf_test_data = torch.cat(encoded[10 * i: 10 * (i + 1)], dim=0)
        clf_test_labels = torch.cat(ys[10 * i: 10 * (i + 1)])

        clf_train = data.TensorDataset(clf_train_data, clf_train_labels)
        clf_test = data.TensorDataset(clf_test_data, clf_test_labels)

        clf = Classifier()

        clf.fit(clf_train)

        test_loader = torch.utils.data.DataLoader(clf_test,
                                                  batch_size=100,
                                                  shuffle=False)

        y_pred, y_true = clf.decision_function_loader(test_loader)

        acc = accuracy_score(y_true, y_pred)

        res.append(acc)
    return res


def load_data(dataset='MNIST'):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])

    if dataset == 'MNIST':
        _ = MNIST(root='./', train=True, transform=transform, download=True)
        _ = MNIST(root='./', train=False, transform=transform, download=True)
    elif dataset == 'CIFAR10':
        _ = CIFAR10(root='./', train=True, transform=transform, download=True)
        _ = CIFAR10(root='./', train=False, transform=transform, download=True)


def get_data(dataset='MNIST'):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])

    if dataset == 'MNIST':
        train_set = MNIST(root='./', train=True, transform=transform, download=True)
        test_set = MNIST(root='./', train=False, transform=transform, download=True)
    elif dataset == 'CIFAR10':
        train_set = CIFAR10(root='./', train=True, transform=transform, download=True)
        test_set = CIFAR10(root='./', train=False, transform=transform, download=True)
    else:
        raise ValueError()

    return train_set, test_set


def plot_training(losses, title='Losses'):
    plt.figure()
    x = np.arange(len(losses))

    plt.plot(x, losses, label='loss')

    plt.legend()
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')


def q1_results(q1, accuracy=False):
    train_data, test_data = get_data('MNIST')
    losses, encoder = q1(train_data)

    plot_training(losses)
    if accuracy:
        acc = test_classification(test_data, encoder)
        print(f'mean classification accuracy={np.mean(acc):.4f}')


def q2_results(q2, accuracy=False):
    train_data, test_data = get_data('CIFAR10')
    losses, encoder = q2(train_data, test_data)

    plot_training(losses)
    if accuracy:
        acc = test_classification(test_data, encoder)
        print(f'mean classification accuracy={np.mean(acc):.4f}')
