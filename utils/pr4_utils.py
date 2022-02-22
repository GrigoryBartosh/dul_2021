from scipy.stats import norm
from sklearn.datasets import make_moons

from .utils import *


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


def load_half_moons(n):
    return make_moons(n_samples=n, noise=0.1)


def q1_sample_data_1():
    train_data, train_labels = load_smiley_face(2000)
    test_data, test_labels = load_smiley_face(1000)
    return train_data, train_labels, test_data, test_labels


def q1_sample_data_2():
    train_data, train_labels = load_half_moons(2000)
    test_data, test_labels = load_half_moons(1000)
    return train_data, train_labels, test_data, test_labels


def get_data(dset_id=1):
    if dset_id == 1:
        train_data, train_labels, test_data, test_labels = q1_sample_data_1()
    else:
        train_data, train_labels, test_data, test_labels = q1_sample_data_2()
    return Dataset(train_data, train_labels), Dataset(test_data, test_labels)


def visualize_q1_data(dset_type):
    if dset_type == 1:
        train_data, train_labels, test_data, test_labels = q1_sample_data_1()
    elif dset_type == 2:
        train_data, train_labels, test_data, test_labels = q1_sample_data_2()
    else:
        raise Exception('Invalid dset_type:', dset_type)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.8))
    ax1.set_title('Train Data')
    ax1.scatter(train_data[:, 0], train_data[:, 1], s=1, c=train_labels)
    ax1.set_xlabel('x1')
    ax1.set_xlabel('x2')
    ax2.set_title('Test Data')
    ax2.scatter(test_data[:, 0], test_data[:, 1], s=1, c=test_labels)
    ax1.set_xlabel('x1')
    ax1.set_xlabel('x2')
    print(f'Dataset {dset_type}')
    plt.show()


def generate_1d_data(n):
    assert n % 2 == 0
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n // 2,))
    gaussian2 = np.random.normal(loc=0.5, scale=0.5, size=(n // 2,))
    labels = [1] * len(gaussian1) + [0] * len(gaussian1)
    return np.concatenate([gaussian1, gaussian2]), np.array(labels)


def load_demo_1(n_train, n_test, visualize=True, train_only=False):
    # 1d distribution, mixture of two gaussians
    train_data, train_labels = generate_1d_data(n_train)
    test_data, test_labels = generate_1d_data(n_test)

    if visualize:
        plt.figure()
        x = np.linspace(-3, 3, num=100)
        densities = 0.5 * norm.pdf(x, loc=-1, scale=0.25) + 0.5 * norm.pdf(x, loc=0.5, scale=0.5)
        plt.figure()
        plt.plot(x, densities)
        plt.show()
        plt.figure()
        plt.hist(train_data, bins=50)
        # plot_hist(train_data, bins=50, title='Train Set')
        plt.show()

    train_dset = Dataset(train_data, train_labels)
    test_dset = Dataset(test_data, test_labels)
    return train_dset, test_dset


class Dataset:
    def __init__(self, data, target):
        self.x = data
        self.y = target

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def plot_train_curves(epochs, train_losses, test_losses, title='', y_label='CE'):
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label='train')
    plt.plot(x_test, test_losses, label='test')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.show()
