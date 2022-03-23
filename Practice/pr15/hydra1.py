import hydra


@hydra.main(config_path="./conf", config_name="config")
def f(cfg):
    print(cfg.node.x)
    print(cfg.node.y)


if __name__ == '__main__':
    f()
