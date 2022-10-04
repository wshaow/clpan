import argparse
import random
import torch

from utils import options as option
from data import create_dataloader, create_dataset


def parse_options(option_file_path):
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(option_file_path)
    return opt


def set_random_seed(opt):
    seed = opt['solver']['manual_seed']
    if seed is None: seed = random.randint(1, 10000)  # 这里尽量直接把seed写死
    print(f"===> Random Seed: [{seed}]")
    random.seed(seed)
    torch.manual_seed(seed)


def get_dataloader(opt):
    train_set, train_loader, val_set, val_loader = None, None, None, None
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt)
            print(f'===> Train Dataset: {train_set.name()}   Number of images: [{len(train_set)}]')
            if train_loader is None: raise ValueError("[Error] The training data does not exist")

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print(f'===> Val Dataset: {val_set.name()}   Number of images: [{len(val_set)}]')

        else:
            raise NotImplementedError(f"[Error] Dataset phase [{phase}] in *.json is not recognized.")
    return train_set, train_loader, val_set, val_loader


def main():
    option_file_path = "./options/train/train_example.json"
    opt = parse_options(option_file_path)

    set_random_seed(opt)


if __name__ == "__main__":
    main()
