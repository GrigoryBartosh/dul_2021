import numpy as np
from sklearn.datasets import make_blobs

from .utils import *


def sample_three_blobs(n):
    centers = np.array([[5, 5], [-5, 5], [0, -5]])
    st_devs = np.array([[1.0, 1.0], [0.2, 0.2], [3.0, 0.5]])
    labels = np.random.randint(0, 3, size=(n,), dtype='int32')
    x = np.random.randn(n, 2) * st_devs[labels] + centers[labels]
    return x.astype('float32')


def sample_four_blobs(n):
    centers = np.array([[5, 5], [5, -5], [-5, -5], [-5, 5]])
    st_devs = [1.0, 1.0, 1.0, 1.0]
    x, _ = make_blobs(n, n_features=2, centers=centers, cluster_std=st_devs,
                      shuffle=True)
    return x.astype('float32')


def sample_smiley_data(n):
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
    return data_x[perm].astype('float32'), data_y[perm]


def sample_diag_guass_data(count):
    rand = np.random.RandomState(0)
    return ([[1.0, 2.0]] + rand.randn(count, 2) * [[5.0, 1.0]]).astype('float32')


def sample_cov_gauss_data(count):
    rand = np.random.RandomState(0)
    return ([[1.0, 2.0]] + (rand.randn(count, 2) * [[5.0, 1.0]]).dot(
        [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2]])).astype('float32')

import numpy as np
import torch
import torch.nn.functional as F


def kl(z, dist1, dist2):
    if isinstance(dist1, Normal) and isinstance(dist2, Normal):
        mu1, log_var1 = dist1.get_mu_std()
        mu2, log_var2 = dist2.get_mu_std()
        kl = 0.5 * log_var2 - 0.5 * log_var1 - 0.5
        kl = kl + (torch.exp(log_var1) + (mu1 - mu2) ** 2) * 0.5 * torch.exp(-log_var2)
        return kl.view(kl.shape[0], -1).sum(-1)
    else:
        return dist1.log_prob(z) - dist2.log_prob(z)


class Distribution(object):
    def __init__(self, params=None):
        self.params = params

    def log_prob(self, x, params=None):
        raise NotImplementedError()

    def expectation(self, params=None):
        raise NotImplementedError()

    def sample(self, params=None):
        raise NotImplementedError()

    def set_params(self, params):
        self.params = params

    def get_params(self, params=None):
        if params is None:
            params = self.params
        assert params is not None
        return params


class Normal(Distribution):
    def __init__(self, params=None, use_mean=False, min_std_dev=None):
        super().__init__(params=params)
        self.use_mean = use_mean
        self.min_std_dev = min_std_dev

    def get_mu_std(self):
        params = self.get_params(None)
        mu, log_var = params.chunk(2, dim=1)
        if self.min_std_dev is not None:
            log_var = torch.log(log_var.exp() + 1 + self.min_std_dev)
        return mu, log_var


    def log_prob(self, x, params=None):
        params = self.get_params(params)
        mu, log_var = params.chunk(2, dim=1)
        if self.min_std_dev is not None:
            log_var = torch.log(log_var.exp() + 1 + self.min_std_dev)

        if self.use_mean:
            return -F.mse_loss(mu, x, reduction='none').view(x.shape[0], -1).sum(-1)

        log_prob = 0.5 * np.log(2 * np.pi)
        log_prob = log_prob + 0.5 * log_var + (x - mu) ** 2 * torch.exp(-log_var) * 0.5
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(-1)
        return -log_prob

    def expectation(self, params=None):
        params = self.get_params(params)
        return params.chunk(2, dim=1)[0]

    def sample(self, params=None):
        params = self.get_params(params)
        mu, log_var = params.chunk(2, dim=1)
        eps = torch.randn_like(mu)
        return mu + eps * (0.5 * log_var).exp()

class Bernoulli(Distribution):
    def log_prob(self, x, params=None):
        params = self.get_params(params)
        return -F.binary_cross_entropy_with_logits(params, x, reduction='none').view(x.shape[0], -1).sum(-1)

    def expectation(self, params=None):
        return torch.sigmoid(self.get_params(params))

    def sample(self, params=None):
        params = self.get_params(params)
        return torch.bernoulli(torch.sigmoid(params))


def get_dist_output_size(dist, var_shape, flattened=False):
    if flattened or isinstance(var_shape, int) or len(var_shape) == 1:
        flattened_size = np.prod(var_shape)
        if isinstance(dist, Normal):
            return (2 * flattened_size,)
        elif isinstance(dist, Bernoulli):
            return (flattened_size,)
        else:
            raise Exception('Invalid dist')
    else:
        assert len(var_shape) == 3
        if isinstance(dist, Normal):
            return (var_shape[0] * 2,) + var_shape[1:]
        elif isinstance(dist, Bernoulli):
            return var_shape

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from torchvision.utils import make_grid


def plot_hist(data, bins=10, xlabel='x', ylabel='Probability', title='', density=None):
    bins = np.concatenate((np.arange(bins) - 0.5, [bins - 1 + 0.5]))

    plt.figure()
    plt.hist(data, bins=bins, density=True)

    if density:
        plt.plot(density[0], density[1], label='distribution')
        plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_2d_dist(dist, title='Learned Distribution'):
    plt.figure()
    plt.imshow(dist)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x0')
    plt.show()


def plot_train_curves(epochs, train_losses, test_losses, title=''):
    x = np.linspace(0, epochs, len(train_losses))
    plt.figure()
    plt.plot(x, train_losses, label='train_loss')
    if test_losses:
        plt.plot(x, test_losses, label='test_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_scatter_2d(points, title='', labels=None):
    plt.figure()
    if labels is not None:
        plt.scatter(points[:, 0], points[:, 1], c=labels,
                    cmap=mpl.colors.ListedColormap(['red', 'blue', 'green', 'purple']))
    else:
        plt.scatter(points[:, 0], points[:, 1])
    plt.title(title)
    plt.show()


def visualize_batch(batch_tensor, nrow=8, title='', figsize=None):
    batch_tensor = batch_tensor.clamp(min=0, max=1)
    grid_img = make_grid(batch_tensor, nrow=nrow)
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

def plot_vae_training_plot(train_losses, test_losses, title, fname):
    elbo_train, recon_train, kl_train = train_losses[:, 0], train_losses[:, 1], train_losses[:, 2]
    elbo_test, recon_test, kl_test = test_losses[:, 0], test_losses[:, 1], test_losses[:, 2]
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, elbo_train, label='-elbo_train')
    plt.plot(x_train, recon_train, label='recon_loss_train')
    plt.plot(x_train, kl_train, label='kl_loss_train')
    plt.plot(x_test, elbo_test, label='-elbo_test')
    plt.plot(x_test, recon_test, label='recon_loss_test')
    plt.plot(x_test, kl_test, label='kl_loss_test')

    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    savefig(fname)


def sample_data_1_a(count):
    rand = np.random.RandomState(0)
    return [[1.0, 2.0]] + (rand.randn(count, 2) * [[5.0, 1.0]]).dot(
        [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2]])


def sample_data_2_a(count):
    rand = np.random.RandomState(0)
    return [[-1.0, 2.0]] + (rand.randn(count, 2) * [[1.0, 5.0]]).dot(
        [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2]])


def sample_data_1_b(count):
    rand = np.random.RandomState(0)
    return [[1.0, 2.0]] + rand.randn(count, 2) * [[5.0, 1.0]]


def sample_data_2_b(count):
    rand = np.random.RandomState(0)
    return [[-1.0, 2.0]] + rand.randn(count, 2) * [[1.0, 5.0]]


def q1_sample_data(part, dset_id):
    assert dset_id in [1, 2]
    assert part in ['a', 'b']
    if part == 'a':
        if dset_id == 1:
            dset_fn = sample_data_1_a
        else:
            dset_fn = sample_data_2_a
    else:
        if dset_id == 1:
            dset_fn = sample_data_1_b
        else:
            dset_fn = sample_data_2_b

    train_data, test_data = dset_fn(10000), dset_fn(2500)
    return train_data.astype('float32'), test_data.astype('float32')


def visualize_q1_data(part, dset_id):
    train_data, test_data = q1_sample_data(part, dset_id)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Train Data')
    ax1.scatter(train_data[:, 0], train_data[:, 1])
    ax2.set_title('Test Data')
    ax2.scatter(test_data[:, 0], test_data[:, 1])
    print(f'Dataset {dset_id}')
    plt.show()


def q1_save_results(part, dset_id, fn):
    train_data, test_data = q1_sample_data(part, dset_id)
    train_losses, test_losses, samples_noise, samples_nonoise = fn(train_data, test_data, part, dset_id)
    print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')

    plot_vae_training_plot(train_losses, test_losses, f'Q1({part}) Dataset {dset_id} Train Plot',
                           f'results/q1_{part}_dset{dset_id}_train_plot.png')
    save_scatter_2d(samples_noise, title='Samples with Decoder Noise',
                    fname=f'results/q1_{part}_dset{dset_id}_sample_with_noise.png')
    save_scatter_2d(samples_nonoise, title='Samples without Decoder Noise',
                    fname=f'results/q1_{part}_dset{dset_id}_sample_without_noise.png')
