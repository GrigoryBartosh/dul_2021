from .utils import *


# borrow from https://github.com/rll/deepul

# Question 1
def pr1_sample_data_1():
    count = 1000
    rand = np.random.RandomState(0)
    samples = 0.4 + 0.1 * rand.randn(count)
    data = np.digitize(samples, np.linspace(0.0, 1.0, 20))
    split = int(0.8 * len(data))
    train_data, test_data = data[:split], data[split:]
    return train_data, test_data


def pr1_sample_data_2():
    count = 10000
    rand = np.random.RandomState(0)
    a = 0.3 + 0.1 * rand.randn(count)
    b = 0.8 + 0.05 * rand.randn(count)
    mask = rand.rand(count) < 0.5
    samples = np.clip(a * mask + b * (1 - mask), 0.0, 1.0)
    data = np.digitize(samples, np.linspace(0.0, 1.0, 100))
    split = int(0.8 * len(data))
    train_data, test_data = data[:split], data[split:]
    return train_data, test_data


def visualize_pr1_data(dset_type):
    if dset_type == 1:
        train_data, test_data = pr1_sample_data_1()
        d = 20
    elif dset_type == 2:
        train_data, test_data = pr1_sample_data_2()
        d = 100
    else:
        raise Exception('Invalid dset_type:', dset_type)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Train Data')
    ax1.hist(train_data, bins=np.arange(d) - 0.5, density=True)
    ax1.set_xlabel('x')
    ax2.set_title('Test Data')
    ax2.hist(test_data, bins=np.arange(d) - 0.5, density=True)
    print(f'Dataset {dset_type}')
    plt.show()


def pr1_save_results(dset_type, part, fn):
    if dset_type == 1:
        train_data, test_data = pr1_sample_data_1()
        d = 20
    elif dset_type == 2:
        train_data, test_data = pr1_sample_data_2()
        d = 100
    else:
        raise Exception('Invalid dset_type:', dset_type)

    train_losses, test_losses, distribution = fn(train_data, test_data, d, dset_type)
    assert np.allclose(np.sum(distribution), 1), f'Distribution sums to {np.sum(distribution)} != 1'

    print(f'Final Test Loss: {test_losses[-1]:.4f}')

    save_training_plot(train_losses, test_losses, f'Q1({part}) Dataset {dset_type} Train Plot',
                       f'results/q1_{part}_dset{dset_type}_train_plot.png')
    save_distribution_1d(train_data, distribution,
                         f'Q1({part}) Dataset {dset_type} Learned Distribution',
                         f'results/q1_{part}_dset{dset_type}_learned_dist.png')


def plot_distribution(model_cls, k=4):
    train_data, test_data = pr1_sample_data_2()

    model = model_cls(d=100, k=k)
    _, _ = model.fit(train_data, test_data, num_epochs=30, lr=1e-1, batch_size=100)
    d = model.d
    k = model.k
    plt.figure()

    priors = F.softmax(model.pi_logits, dim=0)
    plt.hist(train_data, bins=np.arange(d) - 0.5, label='data', density=True)
    x = np.linspace(-0.5, d - 0.5, 1000)

    for i in range(k):
        y = model.single_prob(i).repeat(1000 // d)
        plt.plot(x, y, label=f'prior={priors[i]:.4f}')

    plt.title('Distribution per mode')
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()
    # plt.show()
