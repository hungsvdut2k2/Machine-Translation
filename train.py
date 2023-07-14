from manager import Manager
from config import NMTConfig


def train(config: NMTConfig):
    manager = Manager(config, is_train=True)
    manager.train()


if __name__ == "__main__":
    config = NMTConfig()
    train(config)
