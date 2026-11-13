import hydra

from utils.workflows import build_index


@hydra.main(version_base=None, config_path="configs", config_name="build")
def main(cfg):
    build_index(cfg)


if __name__ == "__main__":
    main()
