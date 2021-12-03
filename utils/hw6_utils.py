from .utils import *


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


def visualize_colored_shapes():
    data_dir = get_data_dir(6)
    train_data, test_data = load_pickled_data(join(data_dir, 'shapes_colored.pkl'))
    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs]
    show_samples(images, title='Colored Shapes Samples')


def visualize_svhn():
    data_dir = get_data_dir(6)
    train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs]
    show_samples(images, title='SVHN Samples')


def visualize_cifar10():
    data_dir = get_data_dir(6)
    train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))
    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs]
    show_samples(images, title='CIFAR10 Samples')


def q1_save_results(dset_id, fn):
    data_dir = get_data_dir(6)
    if dset_id == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
    else:
        train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))

    train_losses, test_losses, samples, reconstructions, interpolations = fn(train_data, test_data, dset_id)
    samples, reconstructions, interpolations = samples.astype('float32'), reconstructions.astype(
        'float32'), interpolations.astype('float32')
    print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')
    plot_vae_training_plot(train_losses, test_losses, f'Q1 Dataset {dset_id} Train Plot',
                           f'results/q1_dset{dset_id}_train_plot.png')
    show_samples(samples, title=f'Q1 Dataset {dset_id} Samples',
                 fname=f'results/q1_dset{dset_id}_samples.png')
    show_samples(reconstructions, title=f'Q1 Dataset {dset_id} Reconstructions',
                 fname=f'results/q1_dset{dset_id}_reconstructions.png')
    show_samples(interpolations, title=f'Q1 Dataset {dset_id} Interpolations',
                 fname=f'results/q1_dset{dset_id}_interpolations.png')


def q2_save_results(dset_id, fn):
    assert dset_id in [1, 2]
    data_dir = get_data_dir(6)
    if dset_id == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
    else:
        train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))

    vqvae_train_losses, vqvae_test_losses, pixelcnn_train_losses, pixelcnn_test_losses, samples, reconstructions = fn(
        train_data, test_data, dset_id)
    samples, reconstructions = samples.astype('float32'), reconstructions.astype('float32')
    print(f'VQ-VAE Final Test Loss: {vqvae_test_losses[-1]:.4f}')
    print(f'PixelCNN Prior Final Test Loss: {pixelcnn_test_losses[-1]:.4f}')
    save_training_plot(vqvae_train_losses, vqvae_test_losses, f'Q2 Dataset {dset_id} VQ-VAE Train Plot',
                       f'results/q2_dset{dset_id}_vqvae_train_plot.png')
    save_training_plot(pixelcnn_train_losses, pixelcnn_test_losses, f'Q3 Dataset {dset_id} PixelCNN Prior Train Plot',
                       f'results/q2_dset{dset_id}_pixelcnn_train_plot.png')
    show_samples(samples, title=f'Q2 Dataset {dset_id} Samples',
                 fname=f'results/q2_dset{dset_id}_samples.png')
    show_samples(reconstructions, title=f'Q3 Dataset {dset_id} Reconstructions',
                 fname=f'results/q2_dset{dset_id}_reconstructions.png')


def b4_a_save_results(dset_id, fn):
    assert dset_id in [1, 2]
    data_dir = get_data_dir(6)
    if dset_id == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
    else:
        train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))

    vqvae_train_losses, vqvae_test_losses, pixelcnn_train_losses, pixelcnn_test_losses, samples, reconstructions = fn(
        train_data, test_data, dset_id)
    samples, reconstructions = samples.astype('float32'), reconstructions.astype('float32')
    print(f'VQ-VAE Final Test Loss: {vqvae_test_losses[-1]:.4f}')
    print(f'PixelCNN Prior Final Test Loss: {pixelcnn_test_losses[-1]:.4f}')
    save_training_plot(vqvae_train_losses, vqvae_test_losses, f'B4(a) Dataset {dset_id} VQ-VAE Train Plot',
                       f'results/b4_a_dset{dset_id}_vqvae_train_plot.png')
    save_training_plot(pixelcnn_train_losses, pixelcnn_test_losses,
                       f'B4(a) Dataset {dset_id} PixelCNN Prior Train Plot',
                       f'results/b4_a_dset{dset_id}_pixelcnn_train_plot.png')
    show_samples(samples, title=f'B4(a) Dataset {dset_id} Samples',
                 fname=f'results/b4_a_dset{dset_id}_samples.png')
    show_samples(reconstructions, title=f'Q4(a) Dataset {dset_id} Reconstructions',
                 fname=f'results/b4_a_dset{dset_id}_reconstructions.png')


def b4_b_save_results(fn):
    part = 'b'
    data_dir = get_data_dir(6)
    train_data, test_data = load_pickled_data(join(data_dir, 'mnist.pkl'))

    train_losses, test_losses, samples, reconstructions = fn(train_data, test_data)
    samples, reconstructions = samples.astype('float32') * 255, reconstructions.astype('float32') * 255
    print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')
    plot_vae_training_plot(train_losses, test_losses, f'B4({part}) Train Plot',
                           f'results/b4_{part}_train_plot.png')
    show_samples(samples, title=f'b4({part}) Samples',
                 fname=f'results/b4_{part}_samples.png')
    show_samples(reconstructions, title=f'B4({part}) Reconstructions',
                 fname=f'results/b4_{part}_reconstructions.png')
