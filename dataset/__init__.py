from dataset.MNIST import create_mnist_dataset
from dataset.CIFAR import create_cifar10_dataset
from dataset.Custom import create_custom_dataset
from dataset.gaussian import create_gaussian_dataset


def create_dataset(**kwargs):
    if kwargs['Dataset']['dataset']== "mnist":
        return create_mnist_dataset(**kwargs)
    elif kwargs['Dataset']['dataset'] == "cifar":
        return create_cifar10_dataset(**kwargs)
    elif kwargs['Dataset']['dataset'] == "custom":
        return create_custom_dataset(**kwargs)
    elif kwargs['Dataset']['dataset'] == "gaussian":
        return create_gaussian_dataset(**kwargs)
    else:
        raise ValueError(f"dataset except one of {'mnist', 'cifar', 'custom'}, but got {kwargs['Dataset']['dataset']}")