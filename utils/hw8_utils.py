from scipy.stats import norm
from torchvision.utils import make_grid

from .utils import *


def generate_data_q1(n):
    gaussian1 = np.random.normal(loc=-0.5, scale=0.25, size=(n,))
    gaussian2 = np.random.normal(loc=0.5, scale=1, size=(n,))
    return gaussian1, gaussian2


def plot_data_q1():
    plt.figure()
    x = np.linspace(-3, 3, num=100)
    density_nu = norm.pdf(x, loc=-0.5, scale=0.25)
    density_de = norm.pdf(x, loc=0.5, scale=1)
    plt.figure()
    plt.plot(x, density_nu, label='numerator')
    plt.plot(x, density_de, label='denumenator')
    plt.legend()
    plt.show()


def plot_dre(model, title=""):
    with torch.no_grad():
        plt.figure()
        x = np.linspace(-2., 2, num=100)
        density_nu = norm.pdf(x, loc=-0.5, scale=0.25)
        density_de = norm.pdf(x, loc=0.5, scale=1)

        ratio_pred = model.r(torch.FloatTensor(x.reshape(-1, 1))).cpu().numpy()

        plt.figure()

        plt.plot(x, ratio_pred, label='pred ratio')
        plt.plot(x, density_nu / density_de, label='true ratio')

        plt.legend()
        plt.title(title)
        plt.show()


def q1_results(q):
    nu_data, de_data = generate_data_q1(5000)
    ratio_pred = q(nu_data, de_data)

    plt.figure()
    x = np.linspace(-2, 2, num=100)
    density_nu = norm.pdf(x, loc=-0.5, scale=0.25)
    density_de = norm.pdf(x, loc=0.5, scale=1)

    plt.figure()

    plt.plot(x, ratio_pred, label='pred ratio')
    plt.plot(x, density_nu / density_de, label='true ratio')

    plt.legend()
    plt.show()


def plot_avb_training_plot(train_elbo, train_clf, test_elbo, test_clf, title='Losses'):
    plt.figure()
    n_epochs = len(test_elbo) - 1
    x_train = np.linspace(0, n_epochs, len(train_elbo))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_elbo, label='-elbo_train')
    plt.plot(x_train, train_clf, label='classif. loss train')
    plt.plot(x_test, test_elbo, label='-elbo_test')
    plt.plot(x_test, test_clf, label='classif. loss test')

    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')


def tensor_to_image(x):
    x = x.detach().cpu().numpy()
    x = x.transpose(1, 2, 0)
    return x


def q2_results(q):
    data_dir = get_data_dir(1)
    train_data, test_data = load_pickled_data(join(data_dir, 'mnist.pkl'))

    train_elbo, train_clf, test_elbo, test_clf, samples = q(train_data, test_data)

    plot_avb_training_plot(train_elbo, train_clf, test_elbo, test_clf, 'Training Loss')

    img = torch.Tensor(samples)
    img = img.view(-1, 1, 28, 28)
    img = make_grid(img, nrow=10)
    img = tensor_to_image(img)
    img = img.clip(0, 1)

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()