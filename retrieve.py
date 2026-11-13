import hydra

from utils.workflows import retrieve


@hydra.main(version_base=None, config_path="configs", config_name="retrieve")
def main(cfg):
    retrieve(cfg)


if __name__ == "__main__":
    main()
