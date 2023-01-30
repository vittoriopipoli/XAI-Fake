import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import logging

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

_logger = logging.getLogger(__name__)


class FakeDataset(Dataset):
    def __init__(
            self,
            annotations_file,
            transform=None
    ):
        super().__init__()
        self.annotations_file = pd.read_csv(annotations_file)
        self.transform = transform
        logging.info(f"Created Fake dataset with {len(self)} images")

    def __getitem__(self, idx):
        img_path = self.annotations_file.iloc[idx, 0]
        target = self.annotations_file.iloc[idx, 1]

        image = default_loader(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return self.annotations_file.shape[0]


def dataset_splitter(annotations_file, split_size=0.8, transform_train=None, transform_test=None):
    ds = FakeDataset(annotations_file, transform_train)
    train_size = int(split_size * len(ds))
    test_size = len(ds) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, test_size])
    eval_size = int(0.5 * test_size)
    test_size = test_size - eval_size
    test_dataset, eval_dataset = torch.utils.data.random_split(test_dataset, [test_size, eval_size])
    test_dataset.trasform = transform_test
    eval_dataset.transform = transform_test
    return train_dataset, test_dataset, eval_dataset


if __name__ == "__main__":
    ANNOTATION_PATH = "../dataset_sampling/dataset_10000.csv"
    train, eval = dataset_splitter(ANNOTATION_PATH)
    print(train.len())
    print(eval.len())
