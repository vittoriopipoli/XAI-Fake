from datasets import load_dataset
from datasets import Image as ImageFeature
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


def dataset_splitter(annotations_file, split_size=0.8, transform_train=None, transform_test=None, seed=42):
    ds = FakeDataset(annotations_file, transform_train)
    train_size = int(split_size * len(ds))
    test_size = len(ds) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, test_size],
                                                                generator=torch.Generator().manual_seed(seed))
    eval_size = int(0.5 * test_size)
    test_size = test_size - eval_size
    test_dataset, eval_dataset = torch.utils.data.random_split(test_dataset, [test_size, eval_size],
                                                               generator=torch.Generator().manual_seed(seed))
    test_dataset.trasform = transform_test
    eval_dataset.transform = transform_test
    return train_dataset, test_dataset, eval_dataset


class AiorNotDataset():
    def __init__(self, train_transform, test_transform, split_size=0.8, seed=42):
        self.ds = load_dataset('competitions/aiornot', split='train').cast_column('image', ImageFeature()).remove_columns("id")
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.split_size = 1 - split_size
        self.seed = seed

    def transform_hf_train(self, examples):
        examples["image"] = [self.train_transform(image) for image in examples["image"]]
        return examples

    def transform_hf_test(self, examples):
        examples["image"] = [self.test_transform(image) for image in examples["image"]]
        return examples

    def aiornot_dataset(self):
        train_size = int(self.split_size * len(self.ds))
        test_size = len(self.ds) - train_size
        ds = self.ds.train_test_split(test_size=self.split_size, seed=self.seed)
        train_dataset = ds['train']
        test_dataset = ds['test']
        train_dataset.set_transform(self.transform_hf_train)
        test_dataset.set_transform(self.transform_hf_test)

        ds = test_dataset.train_test_split(test_size=0.5, seed=self.seed)
        eval_dataset = ds['train']
        test_dataset = ds['test']
        return train_dataset, test_dataset, eval_dataset


if __name__ == "__main__":
    ANNOTATION_PATH = "../dataset_sampling/dataset_10000.csv"
    train, eval = dataset_splitter(ANNOTATION_PATH)
    print(train.len())
    print(eval.len())
