from datasets import load_dataset
from datasets import Image as ImageFeature
from torchvision import transforms
jitter = transforms.Compose(
    [
         #transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.7),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.ToTensor(),

    ]
)
def transform_hf(examples):
    examples["image"] = [jitter(image) for image in examples["image"]]
    return examples

ds = load_dataset('competitions/aiornot').cast_column('image', ImageFeature())
ds.set_transform(transform_hf)
ds["train"][:10]["image"]
ds["train"][0]["image"]
