from .utils import *


def q1ab_save_results(dset_type, part, fn, data_dir: str):
    data_dir = data_dir #get_data_dir(1)
    if dset_type == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'shapes_colored.pkl'))
        img_shape = (20, 20, 3)
    elif dset_type == 2:
        train_data, test_data = load_pickled_data(join(data_dir, 'mnist_colored.pkl'))
        img_shape = (28, 28, 3)
    else:
        raise Exception()

    train_losses, test_losses, samples = fn(train_data, test_data, img_shape, dset_type)
    samples = samples.astype('float32') / 3 * 255

    print(f'Final Test Loss: {test_losses[-1]:.4f}')
    save_training_plot(train_losses, test_losses, f'Q1({part}) Dataset {dset_type} Train Plot',
                       f'results/q1_{part}_dset{dset_type}_train_plot.png')
    show_samples(samples, f'results/q1_{part}_dset{dset_type}_samples.png')


def visualize_q1a_data(dset_type, data_dir: str):
    data_dir = data_dir #get_data_dir(1)
    if dset_type == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'shapes_colored.pkl'))
        name = 'Colored Shape'
    elif dset_type == 2:
        train_data, test_data = load_pickled_data(join(data_dir, 'mnist_colored.pkl'))
        name = 'Colored MNIST'
    else:
        raise Exception('Invalid dset type:', dset_type)

    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs].astype('float32') / 3 * 255
    show_samples(images, title=f'{name} Samples')


def q1ab_get_data(dset_type, data_dir: str):
    data_dir = data_dir #get_data_dir(1)
    if dset_type == 1:
        train_data, _ = load_pickled_data(join(data_dir, 'shapes_colored.pkl'))
    elif dset_type == 2:
        train_data, _ = load_pickled_data(join(data_dir, 'mnist_colored.pkl'))
    else:
        raise Exception('Invalid dset type:', dset_type)

    return train_data


def q1c_save_results(dset_type, q1_c, data_dir: str):
    data_dir = data_dir #get_data_dir(1)
    if dset_type == 1:
        train_data, test_data, train_labels, test_labels = load_pickled_data(join(data_dir, 'shapes.pkl'),
                                                                             include_labels=True)
        img_shape, n_classes = (20, 20), 4
    elif dset_type == 2:
        train_data, test_data, train_labels, test_labels = load_pickled_data(join(data_dir, 'mnist.pkl'),
                                                                             include_labels=True)
        img_shape, n_classes = (28, 28), 10
    else:
        raise Exception('Invalid dset type:', dset_type)

    train_losses, test_losses, samples = q1_c(train_data, train_labels, test_data, test_labels, img_shape, n_classes,
                                              dset_type)
    samples = samples.astype('float32') * 255

    print(f'Final Test Loss: {test_losses[-1]:.4f}')
    save_training_plot(train_losses, test_losses, f'Q1(c) Dataset {dset_type} Train Plot',
                       f'results/q1_c_dset{dset_type}_train_plot.png')
    show_samples(samples, f'results/q1_c_dset{dset_type}_samples.png')


def q1c_get_data(dset_type, data_dir: str):
    data_dir = data_dir #get_data_dir(1)
    if dset_type == 1:
        train_data, test_data, train_labels, test_labels = load_pickled_data(join(data_dir, 'shapes.pkl'),
                                                                             include_labels=True)
        img_shape, n_classes = (20, 20), 4
    elif dset_type == 2:
        train_data, test_data, train_labels, test_labels = load_pickled_data(join(data_dir, 'mnist.pkl'),
                                                                             include_labels=True)
        img_shape, n_classes = (28, 28), 10
    else:
        raise Exception('Invalid dset type:', dset_type)

    return train_data, train_labels, img_shape, n_classes


# Bonuses
def b1a_save_results(b1_a, data_dir: str):
    data_dir = data_dir #get_data_dir(1)
    train_data, test_data = load_pickled_data(join(data_dir, 'mnist_colored.pkl'))
    img_shape = (28, 28, 3)
    train_losses, test_losses, samples = b1_a(train_data, test_data, img_shape)
    samples = samples.astype('float32') / 3 * 255
    print(f'Final Test Loss: {test_losses[-1]:.4f}')
    save_training_plot(train_losses, test_losses, f'B1(a) Train Plot',
                       f'results/b1_a_train_plot.png')
    show_samples(samples, f'results/b1_a_samples.png')


def b1b_save_results(b1_b, data_dir: str):
    data_dir = data_dir #get_data_dir(1)
    train_data, test_data = load_pickled_data(join(data_dir, 'mnist_colored.pkl'))
    img_shape = (28, 28, 3)
    train_losses, test_losses, gray_samples, color_samples = b1_b(train_data, test_data, img_shape)
    gray_samples, color_samples = gray_samples.astype('float32'), color_samples.astype('float32')
    gray_samples *= 255
    gray_samples = gray_samples.repeat(3, axis=-1)
    color_samples = color_samples / 3 * 255
    samples = np.stack((gray_samples, color_samples), axis=1).reshape((-1,) + img_shape)

    print(f'Final Test Loss: {test_losses[-1]:.4f}')
    save_training_plot(train_losses, test_losses, f'B1(b) Train Plot',
                       f'results/b1_b_train_plot.png')
    show_samples(samples, f'results/b1_b_samples.png')


def b1ab_get_data(data_dir: str):
    data_dir = data_dir #get_data_dir(1)
    train_data, test_data = load_pickled_data(join(data_dir, 'mnist_colored.pkl'))
    img_shape = (28, 28, 3)
    return train_data, img_shape


def b1c_save_results(b1_c, data_dir: str):
    data_dir = data_dir #get_data_dir(1)
    train_data, test_data = load_pickled_data(join(data_dir, 'mnist.pkl'))
    train_data, test_data = torch.FloatTensor(train_data).permute(0, 3, 1, 2), torch.FloatTensor(test_data).permute(0,
                                                                                                                    3,
                                                                                                                    1,
                                                                                                                    2)
    train_data = F.interpolate(train_data, scale_factor=2, mode='bilinear')
    test_data = F.interpolate(test_data, scale_factor=2, mode='bilinear')
    train_data, test_data = train_data.permute(0, 2, 3, 1).numpy(), test_data.permute(0, 2, 3, 1).numpy()
    train_data, test_data = (train_data > 0.5).astype('uint8'), (test_data > 0.5).astype('uint8')

    train_losses, test_losses, samples = b1_c(train_data, test_data)
    samples = samples.astype('float32') * 255
    print(f'Final Test Loss: {test_losses[-1]:.4f}')
    save_training_plot(train_losses, test_losses, f'B1(c) Train Plot',
                       f'results/b1_c_train_plot.png')
    show_samples(samples, f'results/b1_c_samples.png')


def b1c_get_data(data_dir: str):
    data_dir = data_dir #get_data_dir(1)
    train_data, _ = load_pickled_data(join(data_dir, 'mnist.pkl'))
    train_data = torch.FloatTensor(train_data).permute(0, 3, 1, 2)

    train_data = F.interpolate(train_data, scale_factor=2, mode='bilinear')
    train_data = train_data.permute(0, 2, 3, 1).numpy()
    train_data = (train_data > 0.5).astype('uint8')

    return train_data
