import os.path, datasets, glob

def load_dataset(dataset_name):
    """Load a dataset using Huggingface Datasets.
    
    Has special workarounds for Datasets' propensity to iterate all the files
    in a locally-stored dataset, and a few other weird gotchas with local data
    that won't be tripped up by people storing everything on Huggingface."""
    path = os.path.join("output", dataset_name)

    dataset = datasets.load_dataset(
        path=path,
        name=dataset_name,
        data_files=glob.glob("train_*.json", root_dir=path),
        split="train"
    )

    return dataset