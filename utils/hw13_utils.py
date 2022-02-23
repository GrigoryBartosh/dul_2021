from .utils import *


def visualize_q1_data():
    data_dir = get_data_dir(13)
    train_data, test_data = load_pickled_data(join(data_dir, 'celeb.pkl'))
    name = 'CelebA'

    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs].astype(np.float32) / 3.0 * 255.0
    show_samples(images, title=f'{name} Samples')

def get_q1_data():
    data_dir = get_data_dir(13)
    train_data, test_data = load_pickled_data(join(data_dir, 'celeb.pkl'))
    return train_data, test_data


def q1_save_results(fn, part):
    data_dir = get_data_dir(13)
    train_data, test_data = load_pickled_data(join(data_dir, 'celeb.pkl'))

    train_losses, test_losses, samples, interpolations = fn(train_data, test_data)
    samples = samples.astype('float')
    interpolations = interpolations.astype('float')

    print(f'Final Test Loss: {test_losses[-1]:.4f}')
    save_training_plot(train_losses, test_losses, f'Q1 Dataset Train Plot',
                       f'results/q1_{part}_train_plot.png')
    show_samples(samples * 255.0, f'results/q1_{part}_samples.png')
    show_samples(interpolations * 255.0, f'results/q1_{part}_interpolations.png', nrow=6, title='Interpolations')
