import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.datasets import MNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageDataset(data.Dataset):
    def __init__(self, imgs, targets, img_transform=None):
        super().__init__()
        self.img_transform = img_transform
        self.imgs = imgs
        self.targets = targets

    def __getitem__(self, i):
        img, target = self.imgs[i], self.targets[i]
        img = Image.fromarray(img)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, target

    def __len__(self):
        return self.imgs.shape[0]


def dataset_from_labels(imgs, targets, class_set, transform=None):
    targets = torch.Tensor(targets)
    class_mask = (targets[:, None] == class_set[None, :]).any(dim=-1)
    return ImageDataset(imgs=imgs[class_mask],
                        targets=targets[class_mask],
                        img_transform=transform)


class FewShotBatchSampler(object):

    def __init__(self, dataset_targets, N_way, K_shot, include_query=True, shuffle=True):
        """
        Inputs:
            dataset_targets - PyTorch tensor of the labels of the data elements.
            N_way - Number of classes to sample per batch.
            K_shot - Number of examples to sample per class in the batch.
            include_query - If True, returns batch of size N_way*K_shot*2
        """
        super().__init__()
        self.dataset_targets = dataset_targets
        self.N_way = N_way
        self.K_shot = K_shot
        self.shuffle = shuffle
        self.include_query = include_query
        if self.include_query:
            self.K_shot *= 2
        self.batch_size = self.N_way * self.K_shot

        # Organize examples by class
        self.classes = torch.unique(self.dataset_targets).tolist()
        self.num_classes = len(self.classes)
        self.indices_per_class = {}
        self.batches_per_class = {}
        for c in self.classes:
            self.indices_per_class[c] = torch.where(self.dataset_targets == c)[0]
            self.batches_per_class[c] = len(self.indices_per_class[c]) // self.K_shot

        # Create a list of classes from which we select the N classes per batch
        self.iterations = sum(self.batches_per_class.values()) // self.N_way
        self.class_list = [c for c in self.classes for _ in range(self.batches_per_class[c])]
        if self.shuffle:
            self.shuffle_data()
        else:
            self.class_list = self.classes * min(self.batches_per_class.values())

    def shuffle_data(self):
        for c in self.classes:
            perm = torch.randperm(self.indices_per_class[c].shape[0])
            self.indices_per_class[c] = self.indices_per_class[c][perm]
        random.shuffle(self.class_list)

    def __iter__(self):
        if self.shuffle:
            self.shuffle_data()

        start_index = defaultdict(int)
        for it in range(self.iterations):
            class_batch = self.class_list[it * self.N_way:(it + 1) * self.N_way]  # Select N classes for the batch
            index_batch = []
            for c in class_batch:  # For each class, select the next K examples and add them to the batch
                index_batch.extend(self.indices_per_class[c][start_index[c]:start_index[c] + self.K_shot])
                start_index[c] += self.K_shot
            if self.include_query:
                index_batch = index_batch[::2] + index_batch[1::2]
            yield index_batch

    def __len__(self):
        return self.iterations


def split_batch(imgs, targets):
    support_imgs, query_imgs = imgs.chunk(2, dim=0)
    support_targets, query_targets = targets.chunk(2, dim=0)
    return support_imgs, query_imgs, support_targets, query_targets


def make_data(n_train, n_test, k_train, k_test):
    train_set = MNIST(root='./', train=True, download=True, transform=transforms.ToTensor())
    test_set = MNIST(root='./', train=False, download=True, transform=transforms.ToTensor())

    # Merging original training and test set
    all_images = np.concatenate([train_set.data, test_set.data], axis=0)
    all_targets = np.concatenate([train_set.targets, test_set.targets], axis=0)

    train_classes, test_classes = torch.arange(10 - n_test), torch.arange(10 - n_test, 10)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])

    train_set = dataset_from_labels(all_images, all_targets, train_classes, transform=transform)
    test_set = dataset_from_labels(all_images, all_targets, test_classes, transform=transform)

    train_data_loader = data.DataLoader(train_set,
                                        batch_sampler=FewShotBatchSampler(train_set.targets,
                                                                          include_query=True,
                                                                          N_way=n_train,
                                                                          K_shot=k_train,
                                                                          shuffle=True))

    test_data_loader = data.DataLoader(test_set,
                                       batch_sampler=FewShotBatchSampler(test_set.targets,
                                                                         include_query=False,
                                                                         N_way=n_test,
                                                                         K_shot=k_test,
                                                                         shuffle=False))

    return train_data_loader, test_data_loader, test_set


def show_imgs():
    train_data_loader, _, _ = make_data(4, 3, 4, 4)

    imgs, targets = next(iter(train_data_loader))
    support_imgs, query_imgs, _, _ = split_batch(imgs, targets)
    support_grid = torchvision.utils.make_grid(support_imgs, nrow=4, normalize=True, pad_value=0.9)
    support_grid = support_grid.permute(1, 2, 0)
    query_grid = torchvision.utils.make_grid(query_imgs, nrow=4, normalize=True, pad_value=0.9)
    query_grid = query_grid.permute(1, 2, 0)

    fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    ax[0].imshow(support_grid)
    ax[0].set_title("Support set")
    ax[0].axis('off')
    ax[1].imshow(query_grid)
    ax[1].set_title("Query set")
    ax[1].axis('off')
    plt.suptitle("Few Shot Batch", weight='bold')
    plt.show()
    plt.close()


def test(net, test_data_loader, test_data, samples=10):
    accs = []
    full_loader = data.DataLoader(test_data, batch_size=200, shuffle=False)
    for i, batch in enumerate(test_data_loader):
        if i >= samples:
            break
        y_pred = net.adapt_few_shots(batch, full_loader)
        accs.append((y_pred == test_data.targets.numpy()).mean())
    return np.mean(accs)


def plot_training(losses, title='Losses'):
    plt.figure()
    x = np.arange(len(losses))

    plt.plot(x, losses, label='loss')

    plt.legend()
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')


def q1_results(q1):
    train_data_loader, test_data_loader, test_set = make_data(4, 3, 4, 4)
    losses, model = q1(train_data_loader)

    plot_training(losses)

    print(f"test accuracy={test(model, test_data_loader, test_set, samples=10):.4f}")


def b_results(b):
    train_data_loader, test_data_loader, test_set = make_data(16, 2, 4, 4)
    losses, model = b(train_data_loader)

    plot_training(losses)

    print(f"test accuracy={test(model, test_data_loader, test_set, samples=10):.4f}")
