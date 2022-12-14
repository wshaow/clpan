import torch.utils.data


def create_dataloader(dataset, dataset_opt):
    phase = dataset_opt['phase']
    if phase == 'train':
        batch_size = dataset_opt['batch_size']
        shuffle = True
        num_workers = dataset_opt['n_workers']
    else:
        batch_size = 1
        shuffle = False
        num_workers = 1
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode'].upper()
    if mode == 'LRHR':
        from data.LRHR_dataset import LRHRDataset as D
    elif mode == "NREG":
        from data.NregDataset import NregDataset as D
    else:
        raise NotImplementedError(f"Dataset {mode} is not recognized.")
    dataset = D(dataset_opt)
    print(f'===> {mode} Dataset is created.')
    return dataset
