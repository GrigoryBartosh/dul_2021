from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10


def get_data(dataset='MNIST', train=False):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])

    if dataset == 'MNIST':
        data_set = MNIST(root='./', train=train, transform=transform, download=True)
    elif dataset == 'CIFAR10':
        data_set = CIFAR10(root='./', train=train, transform=transform, download=True)
    else:
        raise ValueError()

    return data_set


class SSDataset:
    def __init__(self, name='MNIST', c=4000, train=True):
        self.data = get_data(name, train)
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


class DoubleDataset:
    def __init__(self, dset1, dset2):
        self.dset1 = dset1
        self.dset2 = dset2
