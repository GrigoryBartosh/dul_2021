import hydra
from hydra.utils import instantiate


@hydra.main(config_path="./conf", config_name="rec")
def make_data(cfg):
    dd = instantiate(cfg.double_dataset)
    print(len(dd.dset1), len(dd.dset2))


if __name__ == '__main__':
    make_data()
