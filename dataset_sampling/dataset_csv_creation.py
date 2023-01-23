import pandas as pd
import torch
import random
import numpy as np
import os

# Seeds initialization for reproducibility
random.seed(22)
np.random.seed(22)
torch.manual_seed(22)


# Function that samples and create dataset csv from the folders
def create_csv_from_path(real_folder, fake_folder, number_of_samples):
    images_list = []
    target = []
    real_images_path = os.listdir(real_folder)
    random_subsample = random.sample(real_images_path, number_of_samples)
    for real in random_subsample:
        images_list.append(os.path.join(fake_folder, real))
        target.append(0)
    i = 0
    for real in random_subsample:
        i = i % 5
        fake_folder_local = os.path.join(fake_folder, real.split(".")[0])
        fake_images = os.listdir(fake_folder_local)
        # fake_path = random.sample(fake_images,1)
        fake_path = os.path.join(fake_folder_local, fake_images[i])
        images_list.append(fake_path)
        target.append(1)
        i += 1
    d = {"image_path": images_list, "target": target}
    df = pd.DataFrame(data=d)
    df.to_csv("dataset_{}.csv".format(number_of_samples), index=False)


if __name__ == "__main__":
    PATH_REAL_IMAGES = "/nas/softechict-nas-2/datasets/coco/train2014"
    PATH_FAKE_IMAGES = "/mnt/beegfs/work/prin_creative/fake_coco/train2014"
    create_csv_from_path(PATH_REAL_IMAGES, PATH_FAKE_IMAGES, 10000)
