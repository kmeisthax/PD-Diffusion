from PDDiffusion.datasets.WikimediaCommons import local_wikimedia
from PDDiffusion.datasets.load import load_dataset

from torchvision import transforms
from PIL import Image

import datasets, os.path, glob

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
        if type(examples["image"][0]) is dict: #Using map functions unloads PIL images
            images = [preprocess(Image.open(image["path"]).convert("RGB")) for image in examples["image"]]
        else:
            images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        
        return {"image": images}
    
    return transform

def append_base_path_fn(base_path):
    def append_base_fn(path):
        if type(path) == str and os.path.exists(base_path): #Path to local file
            if os.path.sep == "/" and "\\" in path and os.path.sep not in path: #Catch datasets exported on Windows & normalize them
                path = os.path.join(*(path.split("\\")))
            elif os.path.sep == "\\" and "/" in path and os.path.sep not in path: #Same for Linux/macOS paths on Windows
                path = os.path.join(*(path.split("/")))
            
            return {
                "image": os.path.join(base_path, path)
            }
        else: #No base path or dataset already has loaded image data
            return {
                "image": path
            }

    return append_base_fn

def load_dataset_with_imagery(dataset_name, image_size):
    """Load a dataset with an image column.
    
    This uses our custom dataset loader, fixing up the image column to actually
    contain trainable images. It also applies a basic image sizing and
    augmentation pipeline."""
    path = os.path.join("output", dataset_name)

    dataset = load_dataset(dataset_name).map(
        append_base_path_fn(path),
        input_columns="image"
    ).cast_column("image", datasets.Image())

    print(f"Dataset loaded with {dataset.num_rows} rows")

    dataset.set_transform(transformer(image_size), columns=["image"], output_all_columns=True)

    return dataset