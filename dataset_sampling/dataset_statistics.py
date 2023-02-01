import pandas as pd
import torch
from data.fake_dataset_dloader import FakeDataset
import torchvision.transforms as transforms


def compute_stats(annotation_file):
    transform = transforms.Compose([transforms.RandomEqualize(p=1),transforms.ToTensor()])
    fake_dataset = FakeDataset(annotation_file, transform)
    psum_real = torch.tensor([0.0, 0.0, 0.0])
    psum_sq_real = torch.tensor([0.0, 0.0, 0.0])
    psum_fake = torch.tensor([0.0, 0.0, 0.0])
    psum_sq_fake = torch.tensor([0.0, 0.0, 0.0])

    len_real = 0
    len_fake = 0
    # loop through images
    for inputs, label in fake_dataset:
        if label == 0:
            psum_real += inputs.sum(axis=[1, 2])
            psum_sq_real += (inputs ** 2).sum(axis=[1, 2])
            len_real += inputs.shape[1] * inputs.shape[2]
        else:
            psum_fake += inputs.sum(axis=[1, 2])
            psum_sq_fake += (inputs ** 2).sum(axis=[1, 2])
            len_fake += inputs.shape[1] * inputs.shape[2]

    # mean and std
    total_mean_real = psum_real / len_real
    total_var_real = (psum_sq_real / len_real) - (total_mean_real ** 2)
    total_std_real = torch.sqrt(total_var_real)

    # output
    print('mean_real: ' + str(total_mean_real))
    print('std_real:  ' + str(total_std_real))

    # mean and std
    total_mean_fake = psum_fake / len_fake
    total_var_fake = (psum_sq_fake / len_fake) - (total_mean_fake ** 2)
    total_std_fake = torch.sqrt(total_var_fake)

    # output
    print('mean_fake: ' + str(total_mean_fake))
    print('std_fake:  ' + str(total_std_fake))


if __name__ == "__main__":
    compute_stats("dataset_10000.csv")
