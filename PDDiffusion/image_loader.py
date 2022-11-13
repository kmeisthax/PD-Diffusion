from PDDiffusion.datasets.WikimediaCommons import local_wikimedia

from torchvision import transforms
from datasets import Dataset

def convert_and_augment_pipeline(image_size):
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

def transformer(image_size):
    preprocess = convert_and_augment_pipeline(image_size)

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"image": images}
    
    return transform

def load_dataset(image_size):
    dataset = Dataset.from_generator(local_wikimedia)

    dataset.set_transform(transformer(image_size), columns=["image"], output_all_columns=True)

    return dataset