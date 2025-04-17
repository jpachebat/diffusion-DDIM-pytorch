import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import yaml

from utils.distributions import DiffusedGaussianMixture

def create_gaussian_dataset(config_gaussian, batch_size, **kwargs):

    # Retrieve values
    means = torch.tensor(config_gaussian['means'], dtype=torch.float32)
    variances = config_gaussian['variance'] * torch.ones(means.shape[0])
    num_components = means.shape[0]  # Number of Gaussian components
    weights = torch.full((num_components,), 1.0 / num_components, dtype=torch.float32)
    # create gmm model
    gmm = DiffusedGaussianMixture(means,
                                  variances,
                                  weights
                                  )

    arr = gmm.sample(config_gaussian['n_sample'])
    dummy_labels = torch.zeros(arr.shape[0], 1)  # or shape (len(X),) if you prefer
    dataset = TensorDataset(arr, dummy_labels)

    loader_params = dict(
        shuffle=kwargs.get("shuffle", True),
        drop_last=kwargs.get("drop_last", True),
        pin_memory=kwargs.get("pin_memory", True),
        num_workers=kwargs.get("num_workers", 4),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, **loader_params)
    # Return both the dataset and the gmm
    return dataloader, gmm