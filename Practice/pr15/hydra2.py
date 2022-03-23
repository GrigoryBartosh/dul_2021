import hydra
from hydra.utils import instantiate


@hydra.main(config_path="./conf", config_name="data")
def make_data(cfg):
    dataset = instantiate(cfg.dataset)
    print(dataset[0][0].shape)


if __name__ == '__main__':
    make_data()
