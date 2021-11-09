from .utils import *


def q1_a_sample_data(image_file, n, d):
    from PIL import Image
    import itertools

    im = Image.open(image_file).resize((d, d)).convert('L')
    im = np.array(im).astype('float32')
    dist = im / im.sum()

    pairs = list(itertools.product(range(d), range(d)))
    idxs = np.random.choice(len(pairs), size=n, replace=True, p=dist.reshape(-1))
    samples = [pairs[i] for i in idxs]

    return dist, np.array(samples)


def get_data_q1_a(dset_type, data_dir: str):
    # data_dir = get_data_dir(1)
    if dset_type == 1:
        n, d = 10000, 25
        true_dist, data = q1_a_sample_data(f'{data_dir}/smiley.jpg', n, d)
    elif dset_type == 2:
        n, d = 100000, 200
        true_dist, data = q1_a_sample_data(f'{data_dir}/geoffrey-hinton.jpg', n, d)
    else:
        raise Exception('Invalid dset_type:', dset_type)
    return data


def visualize_q1a_data(dset_type, data_dir: str):
    # data_dir = get_data_dir(1)
    if dset_type == 1:
        n, d = 10000, 25
        true_dist, data = q1_a_sample_data(f'{data_dir}/smiley.jpg', n, d)
    elif dset_type == 2:
        n, d = 100000, 200
        true_dist, data = q1_a_sample_data(f'{data_dir}/geoffrey-hinton.jpg', n, d)
    else:
        raise Exception('Invalid dset_type:', dset_type)
    split = int(0.8 * len(data))
    train_data, test_data = data[:split], data[split:]

    train_dist, test_dist = np.zeros((d, d)), np.zeros((d, d))
    for i in range(len(train_data)):
        train_dist[train_data[i][0], train_data[i][1]] += 1
    train_dist /= train_dist.sum()

    for i in range(len(test_data)):
        test_dist[test_data[i][0], test_data[i][1]] += 1
    test_dist /= test_dist.sum()

    print(f'Dataset {dset_type}')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Train Data')
    ax1.imshow(train_dist)
    ax1.axis('off')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x0')

    ax2.set_title('Test Data')
    ax2.imshow(test_dist)
    ax2.axis('off')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x0')

    plt.show()


def get_data_q1_b(dset_type, data_dir: str):
    # data_dir = get_data_dir(1)
    if dset_type == 1:
        train_data, test_data = load_pickled_data(f'{data_dir}/shapes.pkl')
        name = 'Shape'
    elif dset_type == 2:
        train_data, test_data = load_pickled_data(f'{data_dir}/mnist.pkl')
        name = 'MNIST'
    else:
        raise Exception('Invalid dset type:', dset_type)
    return train_data


def visualize_q1b_data(dset_type, data_dir: str):
    # data_dir = get_data_dir(1)
    if dset_type == 1:
        train_data, test_data = load_pickled_data(f'{data_dir}/shapes.pkl')
        name = 'Shape'
    elif dset_type == 2:
        train_data, test_data = load_pickled_data(f'{data_dir}/mnist.pkl')
        name = 'MNIST'
    else:
        raise Exception('Invalid dset type:', dset_type)

    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs] * 255
    show_samples(images, title=f'{name} Samples')


def q1_save_results(dset_type, part, fn, data_dir: str):
    # data_dir = get_data_dir(1)
    if part == 'a':
        if dset_type == 1:
            n, d = 10000, 25
            true_dist, data = q1_a_sample_data(f'{data_dir}/smiley.jpg', n, d)
        elif dset_type == 2:
            n, d = 100000, 200
            true_dist, data = q1_a_sample_data(f'{data_dir}/geoffrey-hinton.jpg', n, d)
        else:
            raise Exception('Invalid dset_type:', dset_type)
        split = int(0.8 * len(data))
        train_data, test_data = data[:split], data[split:]
    elif part == 'b':
        if dset_type == 1:
            train_data, test_data = load_pickled_data(f'{data_dir}/shapes.pkl')
            img_shape = (20, 20)
        elif dset_type == 2:
            train_data, test_data = load_pickled_data(f'{data_dir}/mnist.pkl')
            img_shape = (28, 28)
        else:
            raise Exception('Invalid dset type:', dset_type)
    else:
        raise Exception('Invalid part', part)

    if part == 'a':
        train_losses, test_losses, distribution = fn(train_data, test_data, d, dset_type)
        assert np.allclose(np.sum(distribution), 1), f'Distribution sums to {np.sum(distribution)} != 1'

        print(f'Final Test Loss: {test_losses[-1]:.4f}')

        save_training_plot(train_losses, test_losses, f'Q2({part}) Dataset {dset_type} Train Plot)',
                           f'results/q2_{part}_dset{dset_type}_train_plot.png')
        save_distribution_2d(true_dist, distribution,
                             f'results/q2_{part}_dset{dset_type}_learned_dist.png')
    elif part == 'b':
        train_losses, test_losses, samples = fn(train_data, test_data, img_shape, dset_type)
        samples = samples.astype('float32') * 255
        print(f'Final Test Loss: {test_losses[-1]:.4f}')
        save_training_plot(train_losses, test_losses, f'Q2({part}) Dataset {dset_type} Train Plot',
                           f'results/q2_{part}_dset{dset_type}_train_plot.png')
        show_samples(samples, f'results/q2_{part}_dset{dset_type}_samples.png')
