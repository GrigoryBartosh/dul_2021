from .utils import *


# Question 3
def pr1_save_results(dset_type, pr1):
    data_dir = get_data_dir(1)
    if dset_type == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'shapes.pkl'))
        img_shape = (20, 20)
    elif dset_type == 2:
        train_data, test_data = load_pickled_data(join(data_dir, 'mnist.pkl'))
        img_shape = (28, 28)
    else:
        raise Exception()

    train_losses, test_losses, samples = pr1(train_data, test_data, img_shape, dset_type)
    samples = samples.astype('float32') * 255

    print(f'Final Test Loss: {test_losses[-1]:.4f}')
    save_training_plot(train_losses, test_losses, f'PR1 Dataset {dset_type} Train Plot',
                       f'results/pr1_dset{dset_type}_train_plot.png')
    show_samples(samples, f'results/pr1_dset{dset_type}_samples.png')


def pr1_get_data(dset_type):
    data_dir = get_data_dir(1)
    if dset_type == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'shapes.pkl'))
    elif dset_type == 2:
        train_data, test_data = load_pickled_data(join(data_dir, 'mnist.pkl'))
    else:
        raise Exception()

    return train_data
