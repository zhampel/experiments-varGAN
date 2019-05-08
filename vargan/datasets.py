from __future__ import print_function

try:
    import os
    import numpy as np
    
    import tables
    
    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision import datasets
    import torchvision.transforms as transforms
except ImportError as e:
    print(e)
    raise ImportError


class GaussDataset(Dataset):
    """
    Gaussian dataset class
    """
    def __init__(self, file_name='', transform=None):
        """
        Input
        -----
            file_name : string
                        name of file with dataset
            transform : object
                        optional transforms to apply to samples
        """
        self.file_name = file_name
        self.transform = transform
        self.data_file = self.load_file()

    def load_file(self):
        assert os.path.isfile(self.file_name), "Dataset file %s dne. Exiting..."%self.file_name
        data_file = tables.open_file(self.file_name, mode='r')
        return data_file

    def __len__(self):
        return self.data_file.root.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data_file.root.data[idx,:]
        return sample


DATASET_FN_DICT = {'mnist' : datasets.MNIST,
                   'fashion-mnist' : datasets.FashionMNIST,
                   'gauss' : GaussDataset,
                  }


dataset_list = DATASET_FN_DICT.keys()


def get_dataset(dataset_name='mnist'):
    """
    Convenience function for retrieving
    allowed datasets.
    Parameters
    ----------
    name : {'mnist', 'fashion-mnist'}
          Name of dataset
    Returns
    -------
    fn : function
         PyTorch dataset
    """
    if dataset_name in DATASET_FN_DICT:
        fn = DATASET_FN_DICT[dataset_name]
        return fn
    else:
        raise ValueError('Invalid dataset, {}, entered. Must be '
                         'in {}'.format(dataset_name, DATASET_FN_DICT.keys()))



def get_dataloader(dataset_name='mnist', data_dir='', batch_size=64, train_set=True):

    dset = get_dataset(dataset_name)

    if (dataset_name == 'gauss'):
        dataset = dset(file_name=data_dir)
    else:
        dataset = dset(data_dir, train=train_set, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))

    dataloader = torch.utils.data.DataLoader(
                                             dataset,
                                             batch_size=batch_size,
                                             shuffle=True
                                            )

    return dataloader
